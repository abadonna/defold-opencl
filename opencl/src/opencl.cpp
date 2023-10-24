
// Extension lib defines
#define LIB_NAME "OpenCL"
#define MODULE_NAME "opencl"

// include the Defold SDK
#include <dmsdk/sdk.h>

#if defined(DM_PLATFORM_OSX) || defined(DM_PLATFORM_WINDOWS)
//TODO linux and other platforms

#include <CL/cl.h>
#include <chrono>

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

enum BUFFER_TYPE {
    float1,
    uchar1,
    uint1,
    float3,
    uchar3,
    uint3
};

struct buffer_data {
    cl_mem mem;
    BUFFER_TYPE type;
};

struct kernel_data {
    cl_kernel kernel;
    buffer_data* buffers;
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
        if (data->buffers[i].mem != NULL) {
            clReleaseMemObject(data->buffers[i].mem);
        }
    }
    free(data->buffers);
    clReleaseKernel(data->kernel);
    return 0;
}

void AllocBufferData(kernel_data* kd, int idx)
{
    if (kd->buffers == NULL) {
        kd->buffers = (buffer_data*)malloc(sizeof(buffer_data) * (idx + 1));
    }else if (kd->args_count <= idx) {
        kd->buffers = (buffer_data*)realloc(kd->buffers, sizeof(buffer_data) * (idx + 1));
        for (int i = kd->args_count; i < idx + 1; i ++) {
            kd->buffers[i].mem = NULL;
        }
    }else if (kd->buffers[idx].mem != NULL) {
        clReleaseMemObject(kd->buffers[idx].mem);
    }

    kd->args_count = idx + 1 > kd->args_count ? idx + 1 : kd->args_count;
}

static int SetKernelArgNull(lua_State* L)
{
    kernel_data* kd = (kernel_data*)luaL_checkudata(L, 1, "kernel"); 
    int idx = luaL_checkint(L, 2) - 1; 
    size_t size = luaL_checknumber(L, 3); 
    clSetKernelArg(kd->kernel, idx, size, NULL);
    AllocBufferData(kd, idx);
    kd->buffers[idx].mem = NULL;
    return 0;
}

static int SetKernelArgInt(lua_State* L)
{
    kernel_data* kd = (kernel_data*)luaL_checkudata(L, 1, "kernel"); 
    int idx = luaL_checkint(L, 2) - 1; 
    int value = luaL_checkint(L, 3); 
    clSetKernelArg(kd->kernel, idx, sizeof(int), &value);
    AllocBufferData(kd, idx);
    kd->buffers[idx].mem = NULL;
    return 0;
}

static int SetKernelArgFloat(lua_State* L)
{
    kernel_data* kd = (kernel_data*)luaL_checkudata(L, 1, "kernel"); 
    int idx = luaL_checkint(L, 2) - 1; 
    float value = luaL_checknumber(L, 3); 
    clSetKernelArg(kd->kernel, idx, sizeof(float), &value);
    AllocBufferData(kd, idx);
    kd->buffers[idx].mem = NULL;
    return 0;
}

static int SetKernelArgVec3(lua_State* L)
{
    kernel_data* kd = (kernel_data*)luaL_checkudata(L, 1, "kernel"); 
    int idx = luaL_checkint(L, 2) - 1; 

    cl_float3 value;
    
    lua_rawgeti(L, 3, 1);
    value.x = luaL_checknumber(L, -1);
    lua_pop(L, 1);

    lua_rawgeti(L, 3, 2);
    value.y = luaL_checknumber(L, -1);
    lua_pop(L, 1);

    lua_rawgeti(L, 3, 3);
    value.z = luaL_checknumber(L, -1);
    lua_pop(L, 1);
    
    
    clSetKernelArg(kd->kernel, idx, sizeof(cl_float3), &value);
    AllocBufferData(kd, idx);
    kd->buffers[idx].mem = NULL;
    return 0;
}

template <typename T, typename CLT>
void CreateBufferV3(int idx, uint32_t count, uint32_t stride, void* values, kernel_data* kd, cl_mem_flags flags, BUFFER_TYPE type)
{
    T *data =  (T*)values;
    CLT array[count];
    for (uint i = 0; i < count; ++i) {
        array[i].x = data[0];
        array[i].y = data[1];
        array[i].z = data[2];
        data += stride;
    }

    cl_mem buf = clCreateBuffer(*kd->context, flags, sizeof(CLT) * count, array, NULL);
    clSetKernelArg(kd->kernel, idx, sizeof(cl_mem), &buf);

    kd->buffers[idx].mem = buf;
    kd->buffers[idx].type = type;
}

template <typename T>
void CreateBuffer(int idx, uint32_t count, uint32_t stride, uint32_t components, void* values, kernel_data* kd, cl_mem_flags flags, BUFFER_TYPE type)
{
    T array[count * components];
    T *data = (T*)values;
    for (uint i = 0; i < count; ++i) {
        for (int c = 0; c < components; ++c)
        {
            array[i * components + c] = data[c];
        }

        data += stride;
    }

    cl_mem buf = clCreateBuffer(*kd->context, flags, sizeof(T) * count * components, array, NULL);
    clSetKernelArg(kd->kernel, idx, sizeof(cl_mem), &buf);

    kd->buffers[idx].mem = buf;
    kd->buffers[idx].type = type;
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

    cl_mem_flags flags = CL_MEM_COPY_HOST_PTR;

    if (read && !write) {
        flags |= CL_MEM_READ_ONLY;
    }else if (!read && write) {
        flags |= CL_MEM_WRITE_ONLY;
    }else {
        flags |= CL_MEM_READ_WRITE;
    }

    void* values = 0x0;
    uint32_t count = 0;
    uint32_t components = 0;
    uint32_t stride = 0;
    dmBuffer::Result dataResult = dmBuffer::GetStream(input, streamName, (void**)&values, &count, &components, &stride);
    if (dataResult != dmBuffer::RESULT_OK) {
        return DM_LUA_ERROR("can't get stream");
    }

    dmBuffer::ValueType valuetype;
    
    dmBuffer::GetStreamType(input, streamName, &valuetype, &components);
    const char* stype = dmBuffer::GetValueTypeString(valuetype);
    
    //dmLogInfo("buffer %d, %d, %d", count,components, stride );

    AllocBufferData(kd, idx);

    if ((components == 3) && (strcmp(stype,"VALUE_TYPE_UINT8") == 0)) { //create array of uchar3!
        CreateBufferV3<unsigned char, cl_uchar3>(idx, count, stride, values, kd, flags, uchar3);
        
    } else if ((components == 3) && (strcmp(stype,"VALUE_TYPE_UINT32") == 0)) { 
        CreateBufferV3<uint32_t, cl_uint3>(idx, count, stride, values, kd, flags, uint3);

    } else if (components == 3) { //create array of float3!
        CreateBufferV3<float, cl_float3>(idx, count, stride, values, kd, flags, float3);
        
    } else if (strcmp(stype,"VALUE_TYPE_UINT8") == 0) {
        CreateBuffer<cl_uchar>(idx, count, stride, components, values, kd, flags, uchar1);
        
    } else if (strcmp(stype,"VALUE_TYPE_UINT32") == 0) {
        CreateBuffer<cl_uint>(idx, count, stride, components, values, kd, flags, uint1);

    } else {
        CreateBuffer<float>(idx, count, stride, components, values, kd, flags, float1);
    }
    

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
    using namespace std::chrono;
    
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

    steady_clock::time_point t1 = steady_clock::now();

    cl_int status = clEnqueueNDRangeKernel(*kd->queue, kd->kernel, dim, NULL, global_work_size, local, 0, NULL, NULL);
    //dmLogInfo("execution status: %d", status);

    if (status != CL_SUCCESS) {
        return DM_LUA_ERROR("Kernel execution error.");
    }

    clFinish(*kd->queue);

    steady_clock::time_point t2 = steady_clock::now();
    duration<double> time_span = duration_cast< duration<double> >(t2 - t1);
    
    lua_pushnumber(L, time_span.count());

    return 1;
}

template <typename T>
void Read(lua_State* L, cl_command_queue queue, cl_mem buf, size_t count, T* values, uint32_t stride) 
{
    T output[count];

    clEnqueueReadBuffer(queue,
        buf,
        CL_TRUE,
        0,
        sizeof(T) * count,
        output,
        0,
        NULL,
        NULL);

    if (values != NULL) {
        for (int i = 0; i < count; i++) {
            values[0] = output[i];
            values += stride;
        }
        return;
    }

    lua_newtable(L);

    for (int i = 0; i < count; i++) {
        lua_pushnumber(L, output[i]);
        lua_rawseti(L, -2, i + 1);
    }
}

template <typename T, typename CLT>
void ReadVectors(lua_State* L, cl_command_queue queue, cl_mem buf, size_t count, T *values, uint32_t stride) 
{
    CLT output[count];

    clEnqueueReadBuffer(queue,
        buf,
        CL_TRUE,
        0,
        sizeof(CLT) * count,
        output,
        0,
        NULL,
        NULL);

        if (values != NULL) {
            for (uint i = 0; i < count; i++) {
                values[0] = output[i].x;
                values[1] = output[i].y;
                values[2] = output[i].z;

                values += stride;
            }
            return;
        }

        lua_newtable(L);

        for (int i = 0; i < count; i++) {
            lua_newtable(L);

            lua_pushnumber(L, output[i].x);
            lua_rawseti(L, -2, 1);
            lua_pushnumber(L, output[i].y);
            lua_rawseti(L, -2, 2);
            lua_pushnumber(L, output[i].z);
            lua_rawseti(L, -2, 3);

            lua_rawseti(L, -2, i + 1);
        }
    }

static int ReadKernelBuffer(lua_State* L) 
{
    int ret = lua_gettop(L) == 3 ? 1 : 0;
    DM_LUA_STACK_CHECK(L, ret);

    kernel_data* kd = (kernel_data*)luaL_checkudata(L, 1, "kernel");
    int idx = luaL_checkint(L, 2) - 1;
    size_t count = luaL_checkint(L, 3);

    void* values = NULL;
    uint32_t stride = 0;
    
    if (ret == 0) { //return data in dmBuffer
        dmBuffer::HBuffer output = dmScript::CheckBufferUnpack(L, 4);
        dmhash_t streamName = dmScript::CheckHashOrString(L, 5);
  
        dmBuffer::Result dataResult = dmBuffer::GetStream(output, streamName, (void**)&values, NULL, NULL, &stride);
        if (dataResult != dmBuffer::RESULT_OK) {
            return DM_LUA_ERROR("can't get stream in output buffer");
        }
    }

    switch(kd->buffers[idx].type) {
        case float3: 
            ReadVectors<float, cl_float3>(L, *kd->queue, kd->buffers[idx].mem, count, (float*)values, stride);
            break;
        case uchar3: 
            ReadVectors<unsigned char, cl_uchar3>(L, *kd->queue, kd->buffers[idx].mem, count, (unsigned char*)values, stride);
            break;
        case uint3: 
            ReadVectors<uint32_t, cl_uint3>(L, *kd->queue, kd->buffers[idx].mem, count, (uint32_t*)values, stride);
            break;
        case uchar1: 
            Read<unsigned char>(L, *kd->queue, kd->buffers[idx].mem, count, (unsigned char*)values, stride);
            break;
        case uint1: 
            Read<uint32_t>(L, *kd->queue, kd->buffers[idx].mem, count, (uint32_t*)values, stride);
            break;
        case float1: 
            Read<float>(L, *kd->queue, kd->buffers[idx].mem, count, (float*)values, stride);
            break;
    }
   

    return ret;
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
        {"set_arg_int", SetKernelArgInt},
        {"set_arg_float", SetKernelArgFloat},
        {"set_arg_vec3", SetKernelArgVec3},
        {"set_arg_null", SetKernelArgNull},
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
    bool gpu = false;

    if (lua_gettop(L) == 1) {
        gpu = lua_toboolean(L, 1); 
        lua_pop(L, 1);
    }
    
    cl_device_type type = gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_ALL;
    
    cl_uint num_devices;
    cl_int status = clGetDeviceIDs(platform_id, type, 0, NULL, &num_devices);

    cl_device_id* devices = (cl_device_id *)malloc(sizeof(cl_device_id) * num_devices);
    clGetDeviceIDs(platform_id, type, num_devices, devices, NULL);

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

static dmExtension::Result InitializeExtension(dmExtension::Params* params)
{
    // Init Lua
    LuaInit(params->m_L);
    dmLogInfo("Registered %s Extension", MODULE_NAME);

    cl_uint num_platforms;
    cl_int status = clGetPlatformIDs(1, &platform_id, &num_platforms);
    
    return dmExtension::RESULT_OK;
}

#else

static dmExtension::Result InitializeExtension(dmExtension::Params* params)
{
    // Init Lua
    dmLogInfo("Registered %s Extension", MODULE_NAME);
    return dmExtension::RESULT_OK;
}

#endif // platforms

static dmExtension::Result AppInitializeExtension(dmExtension::AppParams* params)
{
    dmLogInfo("AppInitializeMyExtension");
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

