// generated from rosidl_typesupport_cpp/resource/idl__type_support.cpp.em
// with input from crazyflie_interfaces:srv/Stop.idl
// generated code does not contain a copyright notice

#include "cstddef"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "crazyflie_interfaces/srv/detail/stop__functions.h"
#include "crazyflie_interfaces/srv/detail/stop__struct.hpp"
#include "rosidl_typesupport_cpp/identifier.hpp"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_c/type_support_map.h"
#include "rosidl_typesupport_cpp/message_type_support_dispatch.hpp"
#include "rosidl_typesupport_cpp/visibility_control.h"
#include "rosidl_typesupport_interface/macros.h"

namespace crazyflie_interfaces
{

namespace srv
{

namespace rosidl_typesupport_cpp
{

typedef struct _Stop_Request_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _Stop_Request_type_support_ids_t;

static const _Stop_Request_type_support_ids_t _Stop_Request_message_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_cpp",  // ::rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
    "rosidl_typesupport_introspection_cpp",  // ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  }
};

typedef struct _Stop_Request_type_support_symbol_names_t
{
  const char * symbol_name[2];
} _Stop_Request_type_support_symbol_names_t;

#define STRINGIFY_(s) #s
#define STRINGIFY(s) STRINGIFY_(s)

static const _Stop_Request_type_support_symbol_names_t _Stop_Request_message_typesupport_symbol_names = {
  {
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, crazyflie_interfaces, srv, Stop_Request)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, crazyflie_interfaces, srv, Stop_Request)),
  }
};

typedef struct _Stop_Request_type_support_data_t
{
  void * data[2];
} _Stop_Request_type_support_data_t;

static _Stop_Request_type_support_data_t _Stop_Request_message_typesupport_data = {
  {
    0,  // will store the shared library later
    0,  // will store the shared library later
  }
};

static const type_support_map_t _Stop_Request_message_typesupport_map = {
  2,
  "crazyflie_interfaces",
  &_Stop_Request_message_typesupport_ids.typesupport_identifier[0],
  &_Stop_Request_message_typesupport_symbol_names.symbol_name[0],
  &_Stop_Request_message_typesupport_data.data[0],
};

static const rosidl_message_type_support_t Stop_Request_message_type_support_handle = {
  ::rosidl_typesupport_cpp::typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_Stop_Request_message_typesupport_map),
  ::rosidl_typesupport_cpp::get_message_typesupport_handle_function,
  &crazyflie_interfaces__srv__Stop_Request__get_type_hash,
  &crazyflie_interfaces__srv__Stop_Request__get_type_description,
  &crazyflie_interfaces__srv__Stop_Request__get_type_description_sources,
};

}  // namespace rosidl_typesupport_cpp

}  // namespace srv

}  // namespace crazyflie_interfaces

namespace rosidl_typesupport_cpp
{

template<>
ROSIDL_TYPESUPPORT_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<crazyflie_interfaces::srv::Stop_Request>()
{
  return &::crazyflie_interfaces::srv::rosidl_typesupport_cpp::Stop_Request_message_type_support_handle;
}

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_cpp, crazyflie_interfaces, srv, Stop_Request)() {
  return get_message_type_support_handle<crazyflie_interfaces::srv::Stop_Request>();
}

#ifdef __cplusplus
}
#endif
}  // namespace rosidl_typesupport_cpp

// already included above
// #include "cstddef"
// already included above
// #include "rosidl_runtime_c/message_type_support_struct.h"
// already included above
// #include "crazyflie_interfaces/srv/detail/stop__functions.h"
// already included above
// #include "crazyflie_interfaces/srv/detail/stop__struct.hpp"
// already included above
// #include "rosidl_typesupport_cpp/identifier.hpp"
// already included above
// #include "rosidl_typesupport_cpp/message_type_support.hpp"
// already included above
// #include "rosidl_typesupport_c/type_support_map.h"
// already included above
// #include "rosidl_typesupport_cpp/message_type_support_dispatch.hpp"
// already included above
// #include "rosidl_typesupport_cpp/visibility_control.h"
// already included above
// #include "rosidl_typesupport_interface/macros.h"

namespace crazyflie_interfaces
{

namespace srv
{

namespace rosidl_typesupport_cpp
{

typedef struct _Stop_Response_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _Stop_Response_type_support_ids_t;

static const _Stop_Response_type_support_ids_t _Stop_Response_message_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_cpp",  // ::rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
    "rosidl_typesupport_introspection_cpp",  // ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  }
};

typedef struct _Stop_Response_type_support_symbol_names_t
{
  const char * symbol_name[2];
} _Stop_Response_type_support_symbol_names_t;

#define STRINGIFY_(s) #s
#define STRINGIFY(s) STRINGIFY_(s)

static const _Stop_Response_type_support_symbol_names_t _Stop_Response_message_typesupport_symbol_names = {
  {
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, crazyflie_interfaces, srv, Stop_Response)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, crazyflie_interfaces, srv, Stop_Response)),
  }
};

typedef struct _Stop_Response_type_support_data_t
{
  void * data[2];
} _Stop_Response_type_support_data_t;

static _Stop_Response_type_support_data_t _Stop_Response_message_typesupport_data = {
  {
    0,  // will store the shared library later
    0,  // will store the shared library later
  }
};

static const type_support_map_t _Stop_Response_message_typesupport_map = {
  2,
  "crazyflie_interfaces",
  &_Stop_Response_message_typesupport_ids.typesupport_identifier[0],
  &_Stop_Response_message_typesupport_symbol_names.symbol_name[0],
  &_Stop_Response_message_typesupport_data.data[0],
};

static const rosidl_message_type_support_t Stop_Response_message_type_support_handle = {
  ::rosidl_typesupport_cpp::typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_Stop_Response_message_typesupport_map),
  ::rosidl_typesupport_cpp::get_message_typesupport_handle_function,
  &crazyflie_interfaces__srv__Stop_Response__get_type_hash,
  &crazyflie_interfaces__srv__Stop_Response__get_type_description,
  &crazyflie_interfaces__srv__Stop_Response__get_type_description_sources,
};

}  // namespace rosidl_typesupport_cpp

}  // namespace srv

}  // namespace crazyflie_interfaces

namespace rosidl_typesupport_cpp
{

template<>
ROSIDL_TYPESUPPORT_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<crazyflie_interfaces::srv::Stop_Response>()
{
  return &::crazyflie_interfaces::srv::rosidl_typesupport_cpp::Stop_Response_message_type_support_handle;
}

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_cpp, crazyflie_interfaces, srv, Stop_Response)() {
  return get_message_type_support_handle<crazyflie_interfaces::srv::Stop_Response>();
}

#ifdef __cplusplus
}
#endif
}  // namespace rosidl_typesupport_cpp

// already included above
// #include "cstddef"
// already included above
// #include "rosidl_runtime_c/message_type_support_struct.h"
// already included above
// #include "crazyflie_interfaces/srv/detail/stop__functions.h"
// already included above
// #include "crazyflie_interfaces/srv/detail/stop__struct.hpp"
// already included above
// #include "rosidl_typesupport_cpp/identifier.hpp"
// already included above
// #include "rosidl_typesupport_cpp/message_type_support.hpp"
// already included above
// #include "rosidl_typesupport_c/type_support_map.h"
// already included above
// #include "rosidl_typesupport_cpp/message_type_support_dispatch.hpp"
// already included above
// #include "rosidl_typesupport_cpp/visibility_control.h"
// already included above
// #include "rosidl_typesupport_interface/macros.h"

namespace crazyflie_interfaces
{

namespace srv
{

namespace rosidl_typesupport_cpp
{

typedef struct _Stop_Event_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _Stop_Event_type_support_ids_t;

static const _Stop_Event_type_support_ids_t _Stop_Event_message_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_cpp",  // ::rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
    "rosidl_typesupport_introspection_cpp",  // ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  }
};

typedef struct _Stop_Event_type_support_symbol_names_t
{
  const char * symbol_name[2];
} _Stop_Event_type_support_symbol_names_t;

#define STRINGIFY_(s) #s
#define STRINGIFY(s) STRINGIFY_(s)

static const _Stop_Event_type_support_symbol_names_t _Stop_Event_message_typesupport_symbol_names = {
  {
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, crazyflie_interfaces, srv, Stop_Event)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, crazyflie_interfaces, srv, Stop_Event)),
  }
};

typedef struct _Stop_Event_type_support_data_t
{
  void * data[2];
} _Stop_Event_type_support_data_t;

static _Stop_Event_type_support_data_t _Stop_Event_message_typesupport_data = {
  {
    0,  // will store the shared library later
    0,  // will store the shared library later
  }
};

static const type_support_map_t _Stop_Event_message_typesupport_map = {
  2,
  "crazyflie_interfaces",
  &_Stop_Event_message_typesupport_ids.typesupport_identifier[0],
  &_Stop_Event_message_typesupport_symbol_names.symbol_name[0],
  &_Stop_Event_message_typesupport_data.data[0],
};

static const rosidl_message_type_support_t Stop_Event_message_type_support_handle = {
  ::rosidl_typesupport_cpp::typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_Stop_Event_message_typesupport_map),
  ::rosidl_typesupport_cpp::get_message_typesupport_handle_function,
  &crazyflie_interfaces__srv__Stop_Event__get_type_hash,
  &crazyflie_interfaces__srv__Stop_Event__get_type_description,
  &crazyflie_interfaces__srv__Stop_Event__get_type_description_sources,
};

}  // namespace rosidl_typesupport_cpp

}  // namespace srv

}  // namespace crazyflie_interfaces

namespace rosidl_typesupport_cpp
{

template<>
ROSIDL_TYPESUPPORT_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<crazyflie_interfaces::srv::Stop_Event>()
{
  return &::crazyflie_interfaces::srv::rosidl_typesupport_cpp::Stop_Event_message_type_support_handle;
}

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_cpp, crazyflie_interfaces, srv, Stop_Event)() {
  return get_message_type_support_handle<crazyflie_interfaces::srv::Stop_Event>();
}

#ifdef __cplusplus
}
#endif
}  // namespace rosidl_typesupport_cpp

// already included above
// #include "cstddef"
#include "rosidl_runtime_c/service_type_support_struct.h"
#include "rosidl_typesupport_cpp/service_type_support.hpp"
// already included above
// #include "crazyflie_interfaces/srv/detail/stop__struct.hpp"
// already included above
// #include "rosidl_typesupport_cpp/identifier.hpp"
// already included above
// #include "rosidl_typesupport_c/type_support_map.h"
#include "rosidl_typesupport_cpp/service_type_support_dispatch.hpp"
// already included above
// #include "rosidl_typesupport_cpp/visibility_control.h"
// already included above
// #include "rosidl_typesupport_interface/macros.h"

namespace crazyflie_interfaces
{

namespace srv
{

namespace rosidl_typesupport_cpp
{

typedef struct _Stop_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _Stop_type_support_ids_t;

static const _Stop_type_support_ids_t _Stop_service_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_cpp",  // ::rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
    "rosidl_typesupport_introspection_cpp",  // ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  }
};

typedef struct _Stop_type_support_symbol_names_t
{
  const char * symbol_name[2];
} _Stop_type_support_symbol_names_t;
#define STRINGIFY_(s) #s
#define STRINGIFY(s) STRINGIFY_(s)

static const _Stop_type_support_symbol_names_t _Stop_service_typesupport_symbol_names = {
  {
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, crazyflie_interfaces, srv, Stop)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, crazyflie_interfaces, srv, Stop)),
  }
};

typedef struct _Stop_type_support_data_t
{
  void * data[2];
} _Stop_type_support_data_t;

static _Stop_type_support_data_t _Stop_service_typesupport_data = {
  {
    0,  // will store the shared library later
    0,  // will store the shared library later
  }
};

static const type_support_map_t _Stop_service_typesupport_map = {
  2,
  "crazyflie_interfaces",
  &_Stop_service_typesupport_ids.typesupport_identifier[0],
  &_Stop_service_typesupport_symbol_names.symbol_name[0],
  &_Stop_service_typesupport_data.data[0],
};

static const rosidl_service_type_support_t Stop_service_type_support_handle = {
  ::rosidl_typesupport_cpp::typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_Stop_service_typesupport_map),
  ::rosidl_typesupport_cpp::get_service_typesupport_handle_function,
  ::rosidl_typesupport_cpp::get_message_type_support_handle<crazyflie_interfaces::srv::Stop_Request>(),
  ::rosidl_typesupport_cpp::get_message_type_support_handle<crazyflie_interfaces::srv::Stop_Response>(),
  ::rosidl_typesupport_cpp::get_message_type_support_handle<crazyflie_interfaces::srv::Stop_Event>(),
  &::rosidl_typesupport_cpp::service_create_event_message<crazyflie_interfaces::srv::Stop>,
  &::rosidl_typesupport_cpp::service_destroy_event_message<crazyflie_interfaces::srv::Stop>,
  &crazyflie_interfaces__srv__Stop__get_type_hash,
  &crazyflie_interfaces__srv__Stop__get_type_description,
  &crazyflie_interfaces__srv__Stop__get_type_description_sources,
};

}  // namespace rosidl_typesupport_cpp

}  // namespace srv

}  // namespace crazyflie_interfaces

namespace rosidl_typesupport_cpp
{

template<>
ROSIDL_TYPESUPPORT_CPP_PUBLIC
const rosidl_service_type_support_t *
get_service_type_support_handle<crazyflie_interfaces::srv::Stop>()
{
  return &::crazyflie_interfaces::srv::rosidl_typesupport_cpp::Stop_service_type_support_handle;
}

}  // namespace rosidl_typesupport_cpp
