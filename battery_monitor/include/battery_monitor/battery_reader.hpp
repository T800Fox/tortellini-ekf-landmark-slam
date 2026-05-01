// MTRX3760 2025 Project 2: Warehouse Robot DevKit
// File: battery_reader.hpp
// Author: Liam Kerr
// Header for the battery monitor

#ifndef BATTERY_MONITOR__BATTERY_READER_HPP_
#define BATTERY_MONITOR__BATTERY_READER_HPP_

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/battery_state.hpp"
#include "std_msgs/msg/bool.hpp"

class BatteryReader : public rclcpp::Node
{
public:
  BatteryReader();

private:
    void callback(const sensor_msgs::msg::BatteryState::SharedPtr msg) const;
    rclcpp::Subscription<sensor_msgs::msg::BatteryState>::SharedPtr battery_sub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr auto_charge_pub_;

};

#endif  // BATTERY_MONITOR__BATTERY_READER_HPP_
