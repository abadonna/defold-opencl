
// Extension lib defines
#define LIB_NAME "OpenCL"
#define MODULE_NAME "opencl"

// include the Defold SDK
#include <dmsdk/sdk.h>
#include <CL/cl.h>

extern double getCPUTime();  

struct device_data {
    cl_device_id id;
    cl_context context;
    cl_command_queue queue;
};

struct program_data {
    cl_program program;
    cl_command_queue* queue;
    cl_context* context;
};

struct kernel_data {
    cl_kernel kernel;
    cl_mem* buffers;
    cl_uint args_count;
    cl_command_queue* queue;
    cl_context* context;
};

cl_platform_id platform_id;

static int Device_destroy(lua_State* L){
    dmLogInfo("device destroy");
    device_data* data = (device_data*)luaL_checkudata(L, 1, "device");
    clReleaseCommandQueue(data->queue);
    clReleaseContext(data->context);
    return 0;
}

static int Program_destroy(lua_State* L){
    dmLogInfo("program destroy");
    program_data* data = (program_data*)luaL_checkudata(L, 1, "program");
    clReleaseProgram(data->program);
    return 0;
}

static int Kernel_destroy(lua_State* L){
    dmLogInfo("kernel destroy");
    kernel_data* data = (kernel_data*)luaL_checkudata(L, 1, "kernel");
    for (int i = 0; i < data->args_count; i++) {
        clReleaseMemObject(data->buffers[i]);
    }
    free(data->buffers);
    clReleaseKernel(data->kernel);
    return 0;
}

static int SetKernelArgBuffer(lua_State* L)
{
    //TODO: different value types, check strem dmBuffer::GetStreamType/dmBuffer::GetValueTypeString
    DM_LUA_STACK_CHECK(L, 0);

    kernel_data* kd = (kernel_data*)luaL_checkudata(L, 1, "kernel"); 
    int idx = luaL_checkint(L, 2) - 1; 
    dmBuffer::HBuffer input = dmScript::CheckBufferUnpack(L, 3);
    dmhash_t streamName = dmScript::CheckHashOrString(L, 4);

    bool read = lua_toboolean(L, 5);
    bool write = lua_toboolean(L, 6);

    float* values = 0x0;
    uint32_t count = 0;
    uint32_t components = 0;
    uint32_t stride = 0;
    dmBuffer::Result dataResult = dmBuffer::GetStream(input, streamName, (void**)&values, &count, &components, &stride);
    if (dataResult != dmBuffer::RESULT_OK) {
        return DM_LUA_ERROR("can't get stream");
    }
    dmLogInfo("buffer %d, %d, %d", count,components, stride );

    float array[count * components];

    for (int i = 0; i < count; ++i) {
        for (int c = 0; c < components; ++c)
        {
            //dmLogInfo("set %d, %f", i, values[c]);
            array[i * components + c] = values[c];
        }

        values += stride;
    }


    cl_mem_flags flags = CL_MEM_COPY_HOST_PTR;

    if (read && !write) {
        flags |= CL_MEM_READ_ONLY;
    }else if (!read && write) {
        flags |= CL_MEM_WRITE_ONLY;
    }else {
        flags |= CL_MEM_READ_WRITE;
    }

    cl_mem buf = clCreateBuffer(*kd->context, flags, sizeof(cl_float) * count * components, array, NULL);
    clSetKernelArg(kd->kernel, idx, sizeof(cl_mem), &buf);

    if (kd->args_count == 0) {
        kd->buffers = (cl_mem*)malloc(sizeof(cl_mem) * (idx + 1));
    }else if (kd->args_count <= idx) {
        kd->buffers = (cl_mem*)realloc(kd->buffers, sizeof(cl_mem) * (idx + 1));
    }else if (kd->buffers[kd->args_count] != NULL) {
        clReleaseMemObject(kd->buffers[kd->args_count]);
    }

    kd->buffers[idx] = buf;
    kd->args_count = idx + 1 > kd->args_count ? idx + 1 : kd->args_count;

    return 0;
}

void LoadWorkSize(lua_State* L, size_t* array, int index, cl_uint dim)
{
    for (cl_uint i = 1; i <= dim; i ++) {
        lua_rawgeti(L, index, i);
        size_t n = luaL_checknumber(L, -1);
        lua_pop(L, 1);
        array[i-1] = n;
    }
}

static int RunKernel(lua_State* L)
{
    DM_LUA_STACK_CHECK(L, 1);

    kernel_data* kd = (kernel_data*)luaL_checkudata(L, 1, "kernel");
    cl_uint dim = luaL_checknumber(L, 2);

    size_t global_work_size[dim];
    size_t local_work_size[dim];

    bool has_local_size = lua_gettop(L) == 4;

    LoadWorkSize(L, global_work_size, 3, dim);

    size_t* local = NULL;

    if (has_local_size) {
        LoadWorkSize(L, local_work_size, 4, dim);
        local = local_work_size;
    }

    double rtime = getCPUTime();

    cl_int status = clEnqueueNDRangeKernel(*kd->queue, kd->kernel, dim, NULL, global_work_size, local, 0, NULL, NULL);
    //dmLogInfo("execution status: %d", status);

    if (status != CL_SUCCESS) {
        return DM_LUA_ERROR("Kernel execution error.");
    }

    clFinish(*kd->queue);

    rtime = getCPUTime() - rtime;
    lua_pushnumber(L, rtime);

    return 1;
}

static int ReadKernelBuffer(lua_State* L) 
{
    DM_LUA_STACK_CHECK(L, 1);

    kernel_data* kd = (kernel_data*)luaL_checkudata(L, 1, "kernel");
    int idx = luaL_checkint(L, 2) - 1;
    int count = luaL_checkint(L, 3);

    float output[count];

    clEnqueueReadBuffer(*kd->queue,
        kd->buffers[idx],
        CL_TRUE,
        0,
        sizeof(cl_float) * count,
        output,
        0,
        NULL,
        NULL);


        lua_newtable(L);

        for (int i = 0; i < count; i++) {
            lua_pushnumber(L, output[i]);
            lua_rawseti(L, -2, i + 1);
        }

    return 1;
}

static int CreateKernel(lua_State* L)
{
    DM_LUA_STACK_CHECK(L, 1);

    const char* name = luaL_checkstring(L, -1);
    program_data* p = (program_data*)luaL_checkudata(L, 1, "program"); 

    cl_int err;
    cl_kernel kernel = clCreateKernel(p->program, name, &err);

    if (err != 0) {
        return DM_LUA_ERROR("Can't create kernel.");
    }

    kernel_data* data = (kernel_data*)(lua_newuserdata(L, sizeof(kernel_data)));
    data->kernel = kernel;
    data->args_count = 0;
    data->buffers = NULL;
    data->queue = p->queue;
    data->context = p->context;

    luaL_newmetatable(L, "kernel");
    static const luaL_Reg functions[] =
    {
        {"__gc", Kernel_destroy},
        {"set_arg_buffer", SetKernelArgBuffer},
        {"run", RunKernel},
        {"read", ReadKernelBuffer},
        {0, 0}
    };
    luaL_register(L, NULL, functions);
    lua_pushvalue(L, -1);
    lua_setfield(L, -1, "__index");
    lua_setmetatable(L, -2);
    return 1;
}

static int LoadProgram(lua_State* L)
{
    DM_LUA_STACK_CHECK(L, 1);

    const char* source = luaL_checkstring(L, -1);

    device_data* device = NULL;
    if (lua_istable(L, 1)) {
        lua_getfield(L, 1, "id");
        device = (device_data*)luaL_checkudata(L, -1, "device"); 
        lua_pop(L, 1);
    }else {
        dmLogInfo("device id");
        device = (device_data*)luaL_checkudata(L, 1, "device"); 
    }

    if (device->context == NULL) {
        device->context = clCreateContext(NULL, 1, &device->id, NULL, NULL, NULL);
        device->queue = clCreateCommandQueue(device->context, device->id, 0, NULL);
    }

    cl_program program = clCreateProgramWithSource(device->context, 1, (const char **)&source, NULL, NULL);

    cl_int status = clBuildProgram(program, 1, &device->id, NULL, NULL, NULL);

    if(status != CL_SUCCESS) {
        dmLogInfo("clBuildProgram failed: %d", status);
        return DM_LUA_ERROR("Can't load program.");

    }

    program_data* data = (program_data*)(lua_newuserdata(L, sizeof(program_data)));
    data->program = program;
    data->queue = &device->queue;
    data->context = &device->context;

    luaL_newmetatable(L, "program");
    static const luaL_Reg functions[] =
    {
        {"__gc", Program_destroy},
        {"create_kernel", CreateKernel},
        {0, 0}
    };
    luaL_register(L, NULL, functions);
    lua_pushvalue(L, -1);
    lua_setfield(L, -1, "__index");
    lua_setmetatable(L, -2);

    return 1;
}

static int GetDevices(lua_State* L)
{
    cl_uint num_devices;
    cl_int status = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);

    cl_device_id* devices = (cl_device_id *)malloc(sizeof(cl_device_id) * num_devices);
    clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);

    lua_newtable(L);

    size_t  str_info_size;
    char *  str_info;
    cl_uint uint_info;

    for(int j = 0; j < num_devices; ++j) {

        lua_newtable(L);

        static const luaL_Reg f[] =
        {
            {"load_program", LoadProgram},
            {0, 0}
        };
        luaL_register(L, NULL, f);
        
        clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 0, NULL, &str_info_size);
        str_info = (char *)malloc(str_info_size);
        clGetDeviceInfo(devices[j], CL_DEVICE_NAME, str_info_size, str_info, NULL);
        //dmLogInfo("device %d: %s", j, str_info);

        lua_pushstring(L, "name");
        lua_pushstring(L, str_info);
        lua_settable(L, -3);
        
        free(str_info);

        clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(uint_info), &uint_info, NULL);
        //dmLogInfo("compute units: %d", uint_info);

        lua_pushstring(L, "units");
        lua_pushnumber(L, uint_info);
        lua_settable(L, -3);

        /*
        clGetDeviceInfo(devices[j], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(uint_info), &uint_info, NULL);
        //dmLogInfo("max freq: %d",uint_info);

        lua_pushstring(L, "freq");
        lua_pushnumber(L, uint_info);
        lua_settable(L, -3);*/

        clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(uint_info), &uint_info, NULL);

        lua_pushstring(L, "max_work_groups");
        lua_pushnumber(L, uint_info);
        lua_settable(L, -3);


        cl_uint num;
        clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(num), &num, NULL);

        size_t dims[num];
        clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(dims), &dims, NULL);

        lua_pushstring(L, "max_work_items");
        lua_newtable(L);
    
        for (int k = 0; k < num; k++)
        {
            lua_pushnumber(L, dims[k]);
            lua_rawseti(L, -2, k + 1);
        }

        lua_settable(L, -3);

        lua_pushstring(L, "id");
        device_data* data = (device_data*)(lua_newuserdata(L, sizeof(device_data)));
        data->id = devices[j];
        data->context = NULL;
        data->queue = NULL;

        luaL_newmetatable(L, "device");
        static const luaL_Reg functions[] =
        {
            {"__gc", Device_destroy},
            {0, 0}
        };
        luaL_register(L, NULL, functions);
        lua_pushvalue(L, -1);
        lua_setfield(L, -1, "__index");
        lua_setmetatable(L, -2);
        
        
        //lua_pushnumber(L,  j);
        lua_settable(L, -3);

        lua_rawseti(L, 1, j + 1);
    }

    free(devices);
    return 1;
}



// Functions exposed to Lua
static const luaL_reg Module_methods[] =
{
    {"get_devices", GetDevices},
    {0, 0}
};

static void LuaInit(lua_State* L)
{
    int top = lua_gettop(L);

    // Register lua names
    luaL_register(L, MODULE_NAME, Module_methods);

    lua_pop(L, 1);
    assert(top == lua_gettop(L));
}

static dmExtension::Result AppInitializeExtension(dmExtension::AppParams* params)
{
    dmLogInfo("AppInitializeMyExtension");
    return dmExtension::RESULT_OK;
}

static dmExtension::Result InitializeExtension(dmExtension::Params* params)
{
    // Init Lua
    LuaInit(params->m_L);
    dmLogInfo("Registered %s Extension", MODULE_NAME);

    cl_uint num_platforms;
    cl_int status = clGetPlatformIDs(1, &platform_id, &num_platforms);
    
    return dmExtension::RESULT_OK;
}

static dmExtension::Result AppFinalizeExtension(dmExtension::AppParams* params)
{
    //dmLogInfo("AppFinalizeMyExtension");
    return dmExtension::RESULT_OK;
}

static dmExtension::Result FinalizeExtension(dmExtension::Params* params)
{
    //dmLogInfo("FinalizeMyExtension");
    return dmExtension::RESULT_OK;
}

static dmExtension::Result OnUpdateExtension(dmExtension::Params* params)
{
    //dmLogInfo("OnUpdateMyExtension");
    return dmExtension::RESULT_OK;
}

static void OnEventExtension(dmExtension::Params* params, const dmExtension::Event* event)
{
    switch(event->m_Event)
    {
        case dmExtension::EVENT_ID_ACTIVATEAPP:
            dmLogInfo("OnEventExtension - EVENT_ID_ACTIVATEAPP");
            break;
        case dmExtension::EVENT_ID_DEACTIVATEAPP:
            dmLogInfo("OnEventExtension - EVENT_ID_DEACTIVATEAPP");
            break;
        case dmExtension::EVENT_ID_ICONIFYAPP:
            dmLogInfo("OnEventExtension - EVENT_ID_ICONIFYAPP");
            break;
        case dmExtension::EVENT_ID_DEICONIFYAPP:
            dmLogInfo("OnEventExtension - EVENT_ID_DEICONIFYAPP");
            break;
        default:
            dmLogWarning("OnEventExtension - Unknown event id");
            break;
    }
}

// Defold SDK uses a macro for setting up extension entry points:
//
// DM_DECLARE_EXTENSION(symbol, name, app_init, app_final, init, update, on_event, final)

DM_DECLARE_EXTENSION(OpenCL, LIB_NAME, AppInitializeExtension, AppFinalizeExtension, InitializeExtension, NULL, NULL, FinalizeExtension)
