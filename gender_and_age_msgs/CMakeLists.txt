cmake_minimum_required(VERSION 2.8.3)
project(gender_and_age_msgs)

find_package(catkin REQUIRED
  COMPONENTS
    std_msgs
    sensor_msgs
    message_generation
)

add_message_files(
   FILES
    GenderAndAge.msg
    GenderProbability.msg
    AgeProbability.msg
    GenderAndAgeList.msg
)

add_service_files(
  FILES
   GenderAndAgeService.srv
)

 generate_messages(
   DEPENDENCIES
   sensor_msgs
   std_msgs
)

catkin_package(
  CATKIN_DEPENDS
   message_runtime
)
