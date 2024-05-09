// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from crazyflie_interfaces:srv/StartTrajectory.idl
// generated code does not contain a copyright notice

#ifndef CRAZYFLIE_INTERFACES__SRV__DETAIL__START_TRAJECTORY__FUNCTIONS_H_
#define CRAZYFLIE_INTERFACES__SRV__DETAIL__START_TRAJECTORY__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/action_type_support_struct.h"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_runtime_c/service_type_support_struct.h"
#include "rosidl_runtime_c/type_description/type_description__struct.h"
#include "rosidl_runtime_c/type_description/type_source__struct.h"
#include "rosidl_runtime_c/type_hash.h"
#include "rosidl_runtime_c/visibility_control.h"
#include "crazyflie_interfaces/msg/rosidl_generator_c__visibility_control.h"

#include "crazyflie_interfaces/srv/detail/start_trajectory__struct.h"

/// Retrieve pointer to the hash of the description of this type.
ROSIDL_GENERATOR_C_PUBLIC_crazyflie_interfaces
const rosidl_type_hash_t *
crazyflie_interfaces__srv__StartTrajectory__get_type_hash(
  const rosidl_service_type_support_t * type_support);

/// Retrieve pointer to the description of this type.
ROSIDL_GENERATOR_C_PUBLIC_crazyflie_interfaces
const rosidl_runtime_c__type_description__TypeDescription *
crazyflie_interfaces__srv__StartTrajectory__get_type_description(
  const rosidl_service_type_support_t * type_support);

/// Retrieve pointer to the single raw source text that defined this type.
ROSIDL_GENERATOR_C_PUBLIC_crazyflie_interfaces
const rosidl_runtime_c__type_description__TypeSource *
crazyflie_interfaces__srv__StartTrajectory__get_individual_type_description_source(
  const rosidl_service_type_support_t * type_support);

/// Retrieve pointer to the recursive raw sources that defined the description of this type.
ROSIDL_GENERATOR_C_PUBLIC_crazyflie_interfaces
const rosidl_runtime_c__type_description__TypeSource__Sequence *
crazyflie_interfaces__srv__StartTrajectory__get_type_description_sources(
  const rosidl_service_type_support_t * type_support);

/// Initialize srv/StartTrajectory message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * crazyflie_interfaces__srv__StartTrajectory_Request
 * )) before or use
 * crazyflie_interfaces__srv__StartTrajectory_Request__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_crazyflie_interfaces
bool
crazyflie_interfaces__srv__StartTrajectory_Request__init(crazyflie_interfaces__srv__StartTrajectory_Request * msg);

/// Finalize srv/StartTrajectory message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_crazyflie_interfaces
void
crazyflie_interfaces__srv__StartTrajectory_Request__fini(crazyflie_interfaces__srv__StartTrajectory_Request * msg);

/// Create srv/StartTrajectory message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * crazyflie_interfaces__srv__StartTrajectory_Request__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_crazyflie_interfaces
crazyflie_interfaces__srv__StartTrajectory_Request *
crazyflie_interfaces__srv__StartTrajectory_Request__create();

/// Destroy srv/StartTrajectory message.
/**
 * It calls
 * crazyflie_interfaces__srv__StartTrajectory_Request__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_crazyflie_interfaces
void
crazyflie_interfaces__srv__StartTrajectory_Request__destroy(crazyflie_interfaces__srv__StartTrajectory_Request * msg);

/// Check for srv/StartTrajectory message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_crazyflie_interfaces
bool
crazyflie_interfaces__srv__StartTrajectory_Request__are_equal(const crazyflie_interfaces__srv__StartTrajectory_Request * lhs, const crazyflie_interfaces__srv__StartTrajectory_Request * rhs);

/// Copy a srv/StartTrajectory message.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source message pointer.
 * \param[out] output The target message pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer is null
 *   or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_crazyflie_interfaces
bool
crazyflie_interfaces__srv__StartTrajectory_Request__copy(
  const crazyflie_interfaces__srv__StartTrajectory_Request * input,
  crazyflie_interfaces__srv__StartTrajectory_Request * output);

/// Retrieve pointer to the hash of the description of this type.
ROSIDL_GENERATOR_C_PUBLIC_crazyflie_interfaces
const rosidl_type_hash_t *
crazyflie_interfaces__srv__StartTrajectory_Request__get_type_hash(
  const rosidl_message_type_support_t * type_support);

/// Retrieve pointer to the description of this type.
ROSIDL_GENERATOR_C_PUBLIC_crazyflie_interfaces
const rosidl_runtime_c__type_description__TypeDescription *
crazyflie_interfaces__srv__StartTrajectory_Request__get_type_description(
  const rosidl_message_type_support_t * type_support);

/// Retrieve pointer to the single raw source text that defined this type.
ROSIDL_GENERATOR_C_PUBLIC_crazyflie_interfaces
const rosidl_runtime_c__type_description__TypeSource *
crazyflie_interfaces__srv__StartTrajectory_Request__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support);

/// Retrieve pointer to the recursive raw sources that defined the description of this type.
ROSIDL_GENERATOR_C_PUBLIC_crazyflie_interfaces
const rosidl_runtime_c__type_description__TypeSource__Sequence *
crazyflie_interfaces__srv__StartTrajectory_Request__get_type_description_sources(
  const rosidl_message_type_support_t * type_support);

/// Initialize array of srv/StartTrajectory messages.
/**
 * It allocates the memory for the number of elements and calls
 * crazyflie_interfaces__srv__StartTrajectory_Request__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_crazyflie_interfaces
bool
crazyflie_interfaces__srv__StartTrajectory_Request__Sequence__init(crazyflie_interfaces__srv__StartTrajectory_Request__Sequence * array, size_t size);

/// Finalize array of srv/StartTrajectory messages.
/**
 * It calls
 * crazyflie_interfaces__srv__StartTrajectory_Request__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_crazyflie_interfaces
void
crazyflie_interfaces__srv__StartTrajectory_Request__Sequence__fini(crazyflie_interfaces__srv__StartTrajectory_Request__Sequence * array);

/// Create array of srv/StartTrajectory messages.
/**
 * It allocates the memory for the array and calls
 * crazyflie_interfaces__srv__StartTrajectory_Request__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_crazyflie_interfaces
crazyflie_interfaces__srv__StartTrajectory_Request__Sequence *
crazyflie_interfaces__srv__StartTrajectory_Request__Sequence__create(size_t size);

/// Destroy array of srv/StartTrajectory messages.
/**
 * It calls
 * crazyflie_interfaces__srv__StartTrajectory_Request__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_crazyflie_interfaces
void
crazyflie_interfaces__srv__StartTrajectory_Request__Sequence__destroy(crazyflie_interfaces__srv__StartTrajectory_Request__Sequence * array);

/// Check for srv/StartTrajectory message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_crazyflie_interfaces
bool
crazyflie_interfaces__srv__StartTrajectory_Request__Sequence__are_equal(const crazyflie_interfaces__srv__StartTrajectory_Request__Sequence * lhs, const crazyflie_interfaces__srv__StartTrajectory_Request__Sequence * rhs);

/// Copy an array of srv/StartTrajectory messages.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source array pointer.
 * \param[out] output The target array pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer
 *   is null or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_crazyflie_interfaces
bool
crazyflie_interfaces__srv__StartTrajectory_Request__Sequence__copy(
  const crazyflie_interfaces__srv__StartTrajectory_Request__Sequence * input,
  crazyflie_interfaces__srv__StartTrajectory_Request__Sequence * output);

/// Initialize srv/StartTrajectory message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * crazyflie_interfaces__srv__StartTrajectory_Response
 * )) before or use
 * crazyflie_interfaces__srv__StartTrajectory_Response__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_crazyflie_interfaces
bool
crazyflie_interfaces__srv__StartTrajectory_Response__init(crazyflie_interfaces__srv__StartTrajectory_Response * msg);

/// Finalize srv/StartTrajectory message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_crazyflie_interfaces
void
crazyflie_interfaces__srv__StartTrajectory_Response__fini(crazyflie_interfaces__srv__StartTrajectory_Response * msg);

/// Create srv/StartTrajectory message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * crazyflie_interfaces__srv__StartTrajectory_Response__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_crazyflie_interfaces
crazyflie_interfaces__srv__StartTrajectory_Response *
crazyflie_interfaces__srv__StartTrajectory_Response__create();

/// Destroy srv/StartTrajectory message.
/**
 * It calls
 * crazyflie_interfaces__srv__StartTrajectory_Response__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_crazyflie_interfaces
void
crazyflie_interfaces__srv__StartTrajectory_Response__destroy(crazyflie_interfaces__srv__StartTrajectory_Response * msg);

/// Check for srv/StartTrajectory message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_crazyflie_interfaces
bool
crazyflie_interfaces__srv__StartTrajectory_Response__are_equal(const crazyflie_interfaces__srv__StartTrajectory_Response * lhs, const crazyflie_interfaces__srv__StartTrajectory_Response * rhs);

/// Copy a srv/StartTrajectory message.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source message pointer.
 * \param[out] output The target message pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer is null
 *   or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_crazyflie_interfaces
bool
crazyflie_interfaces__srv__StartTrajectory_Response__copy(
  const crazyflie_interfaces__srv__StartTrajectory_Response * input,
  crazyflie_interfaces__srv__StartTrajectory_Response * output);

/// Retrieve pointer to the hash of the description of this type.
ROSIDL_GENERATOR_C_PUBLIC_crazyflie_interfaces
const rosidl_type_hash_t *
crazyflie_interfaces__srv__StartTrajectory_Response__get_type_hash(
  const rosidl_message_type_support_t * type_support);

/// Retrieve pointer to the description of this type.
ROSIDL_GENERATOR_C_PUBLIC_crazyflie_interfaces
const rosidl_runtime_c__type_description__TypeDescription *
crazyflie_interfaces__srv__StartTrajectory_Response__get_type_description(
  const rosidl_message_type_support_t * type_support);

/// Retrieve pointer to the single raw source text that defined this type.
ROSIDL_GENERATOR_C_PUBLIC_crazyflie_interfaces
const rosidl_runtime_c__type_description__TypeSource *
crazyflie_interfaces__srv__StartTrajectory_Response__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support);

/// Retrieve pointer to the recursive raw sources that defined the description of this type.
ROSIDL_GENERATOR_C_PUBLIC_crazyflie_interfaces
const rosidl_runtime_c__type_description__TypeSource__Sequence *
crazyflie_interfaces__srv__StartTrajectory_Response__get_type_description_sources(
  const rosidl_message_type_support_t * type_support);

/// Initialize array of srv/StartTrajectory messages.
/**
 * It allocates the memory for the number of elements and calls
 * crazyflie_interfaces__srv__StartTrajectory_Response__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_crazyflie_interfaces
bool
crazyflie_interfaces__srv__StartTrajectory_Response__Sequence__init(crazyflie_interfaces__srv__StartTrajectory_Response__Sequence * array, size_t size);

/// Finalize array of srv/StartTrajectory messages.
/**
 * It calls
 * crazyflie_interfaces__srv__StartTrajectory_Response__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_crazyflie_interfaces
void
crazyflie_interfaces__srv__StartTrajectory_Response__Sequence__fini(crazyflie_interfaces__srv__StartTrajectory_Response__Sequence * array);

/// Create array of srv/StartTrajectory messages.
/**
 * It allocates the memory for the array and calls
 * crazyflie_interfaces__srv__StartTrajectory_Response__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_crazyflie_interfaces
crazyflie_interfaces__srv__StartTrajectory_Response__Sequence *
crazyflie_interfaces__srv__StartTrajectory_Response__Sequence__create(size_t size);

/// Destroy array of srv/StartTrajectory messages.
/**
 * It calls
 * crazyflie_interfaces__srv__StartTrajectory_Response__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_crazyflie_interfaces
void
crazyflie_interfaces__srv__StartTrajectory_Response__Sequence__destroy(crazyflie_interfaces__srv__StartTrajectory_Response__Sequence * array);

/// Check for srv/StartTrajectory message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_crazyflie_interfaces
bool
crazyflie_interfaces__srv__StartTrajectory_Response__Sequence__are_equal(const crazyflie_interfaces__srv__StartTrajectory_Response__Sequence * lhs, const crazyflie_interfaces__srv__StartTrajectory_Response__Sequence * rhs);

/// Copy an array of srv/StartTrajectory messages.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source array pointer.
 * \param[out] output The target array pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer
 *   is null or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_crazyflie_interfaces
bool
crazyflie_interfaces__srv__StartTrajectory_Response__Sequence__copy(
  const crazyflie_interfaces__srv__StartTrajectory_Response__Sequence * input,
  crazyflie_interfaces__srv__StartTrajectory_Response__Sequence * output);

/// Initialize srv/StartTrajectory message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * crazyflie_interfaces__srv__StartTrajectory_Event
 * )) before or use
 * crazyflie_interfaces__srv__StartTrajectory_Event__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_crazyflie_interfaces
bool
crazyflie_interfaces__srv__StartTrajectory_Event__init(crazyflie_interfaces__srv__StartTrajectory_Event * msg);

/// Finalize srv/StartTrajectory message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_crazyflie_interfaces
void
crazyflie_interfaces__srv__StartTrajectory_Event__fini(crazyflie_interfaces__srv__StartTrajectory_Event * msg);

/// Create srv/StartTrajectory message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * crazyflie_interfaces__srv__StartTrajectory_Event__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_crazyflie_interfaces
crazyflie_interfaces__srv__StartTrajectory_Event *
crazyflie_interfaces__srv__StartTrajectory_Event__create();

/// Destroy srv/StartTrajectory message.
/**
 * It calls
 * crazyflie_interfaces__srv__StartTrajectory_Event__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_crazyflie_interfaces
void
crazyflie_interfaces__srv__StartTrajectory_Event__destroy(crazyflie_interfaces__srv__StartTrajectory_Event * msg);

/// Check for srv/StartTrajectory message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_crazyflie_interfaces
bool
crazyflie_interfaces__srv__StartTrajectory_Event__are_equal(const crazyflie_interfaces__srv__StartTrajectory_Event * lhs, const crazyflie_interfaces__srv__StartTrajectory_Event * rhs);

/// Copy a srv/StartTrajectory message.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source message pointer.
 * \param[out] output The target message pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer is null
 *   or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_crazyflie_interfaces
bool
crazyflie_interfaces__srv__StartTrajectory_Event__copy(
  const crazyflie_interfaces__srv__StartTrajectory_Event * input,
  crazyflie_interfaces__srv__StartTrajectory_Event * output);

/// Retrieve pointer to the hash of the description of this type.
ROSIDL_GENERATOR_C_PUBLIC_crazyflie_interfaces
const rosidl_type_hash_t *
crazyflie_interfaces__srv__StartTrajectory_Event__get_type_hash(
  const rosidl_message_type_support_t * type_support);

/// Retrieve pointer to the description of this type.
ROSIDL_GENERATOR_C_PUBLIC_crazyflie_interfaces
const rosidl_runtime_c__type_description__TypeDescription *
crazyflie_interfaces__srv__StartTrajectory_Event__get_type_description(
  const rosidl_message_type_support_t * type_support);

/// Retrieve pointer to the single raw source text that defined this type.
ROSIDL_GENERATOR_C_PUBLIC_crazyflie_interfaces
const rosidl_runtime_c__type_description__TypeSource *
crazyflie_interfaces__srv__StartTrajectory_Event__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support);

/// Retrieve pointer to the recursive raw sources that defined the description of this type.
ROSIDL_GENERATOR_C_PUBLIC_crazyflie_interfaces
const rosidl_runtime_c__type_description__TypeSource__Sequence *
crazyflie_interfaces__srv__StartTrajectory_Event__get_type_description_sources(
  const rosidl_message_type_support_t * type_support);

/// Initialize array of srv/StartTrajectory messages.
/**
 * It allocates the memory for the number of elements and calls
 * crazyflie_interfaces__srv__StartTrajectory_Event__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_crazyflie_interfaces
bool
crazyflie_interfaces__srv__StartTrajectory_Event__Sequence__init(crazyflie_interfaces__srv__StartTrajectory_Event__Sequence * array, size_t size);

/// Finalize array of srv/StartTrajectory messages.
/**
 * It calls
 * crazyflie_interfaces__srv__StartTrajectory_Event__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_crazyflie_interfaces
void
crazyflie_interfaces__srv__StartTrajectory_Event__Sequence__fini(crazyflie_interfaces__srv__StartTrajectory_Event__Sequence * array);

/// Create array of srv/StartTrajectory messages.
/**
 * It allocates the memory for the array and calls
 * crazyflie_interfaces__srv__StartTrajectory_Event__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_crazyflie_interfaces
crazyflie_interfaces__srv__StartTrajectory_Event__Sequence *
crazyflie_interfaces__srv__StartTrajectory_Event__Sequence__create(size_t size);

/// Destroy array of srv/StartTrajectory messages.
/**
 * It calls
 * crazyflie_interfaces__srv__StartTrajectory_Event__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_crazyflie_interfaces
void
crazyflie_interfaces__srv__StartTrajectory_Event__Sequence__destroy(crazyflie_interfaces__srv__StartTrajectory_Event__Sequence * array);

/// Check for srv/StartTrajectory message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_crazyflie_interfaces
bool
crazyflie_interfaces__srv__StartTrajectory_Event__Sequence__are_equal(const crazyflie_interfaces__srv__StartTrajectory_Event__Sequence * lhs, const crazyflie_interfaces__srv__StartTrajectory_Event__Sequence * rhs);

/// Copy an array of srv/StartTrajectory messages.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source array pointer.
 * \param[out] output The target array pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer
 *   is null or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_crazyflie_interfaces
bool
crazyflie_interfaces__srv__StartTrajectory_Event__Sequence__copy(
  const crazyflie_interfaces__srv__StartTrajectory_Event__Sequence * input,
  crazyflie_interfaces__srv__StartTrajectory_Event__Sequence * output);
#ifdef __cplusplus
}
#endif

#endif  // CRAZYFLIE_INTERFACES__SRV__DETAIL__START_TRAJECTORY__FUNCTIONS_H_
