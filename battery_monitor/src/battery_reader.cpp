// MTRX3760 2025 Project 2: Warehouse Robot DevKit
// File: battery_reader.cpp
// Author: Liam Kerr

// Reports the battery level, including voltage and percent
// Gives a warning when the battery level is below 50%
// Publishes to auto_charge topic when below 40% to return to docking station via nav2

// run with
// $ ros2 run battery_monitor battery_reader

// test low battery with:
// $ ros2 topic pub /battery_state sensor_msgs/msg/BatteryState "{voltage: 10.5, percentage: 35.0}" -r 20
#include "battery_monitor/battery_reader.hpp"

const int CHARGED_BATTERY_PERCENT = 90;
const int LOW_BATTERY_PERCENT = 50;        // charging is recommended when under this value
const int CRITICAL_BATTERY_PERCENT = 30;   // battery should be immediately charged when under this value to prevent LIPO damage
const double TIME_FACTOR = 0.5;           // scale factor to convert battery percentage into minutes of operation remaining

BatteryReader::BatteryReader() : Node("battery_reader")
{
    battery_sub_ = this->create_subscription<sensor_msgs::msg::BatteryState>(
        "/battery_state", 10,
        std::bind(&BatteryReader::callback, this, std::placeholders::_1));

    auto_charge_pub_ = this->create_publisher<std_msgs::msg::Bool>("auto_charge", 10);
}

void BatteryReader::callback(const sensor_msgs::msg::BatteryState::SharedPtr msg) const
{
    double voltage = msg->voltage;
    double percent = msg->percentage;
    bool auto_charge_active_;

    double time_est = percent*TIME_FACTOR; // battery life estimate in minutes

    // Latch auto_charge
    // since the battery level can fluxuate by about a percent
    // this prevents inconsistent descision making at the threshold
    // and ensures the destination set by nav2 will not be inconsistent
    if (percent < CRITICAL_BATTERY_PERCENT)
    {    
        auto_charge_active_ = true;
    }
        // additionally ensures the bot is fully charged before leaaving
        // useful for charging setups where the bot does not have to be manually plugged in
    else if (percent > CHARGED_BATTERY_PERCENT) 
    {
        auto_charge_active_ = false;
    }

    std_msgs::msg::Bool auto_charge_msg;
    auto_charge_msg.data = auto_charge_active_;
    auto_charge_pub_->publish(auto_charge_msg);


    if (percent < CRITICAL_BATTERY_PERCENT)
    {
        RCLCPP_WARN(this->get_logger(),
            "CRITICAL BATTERY WARNING, NAVIGATING TO CHARGING STATION: Voltage %.2f V | Percentage %.1f%% | %.2f Minutes of Operation Remaining",
            voltage, percent, time_est);
    }
    else if (percent < LOW_BATTERY_PERCENT)
    {
        RCLCPP_INFO(this->get_logger(),
            "LOW BATTERY WARNING: Voltage %.2f V | Percentage %.1f%% | %.2f Minutes of Operation Remaining",
            voltage, percent, time_est);
    }
    else
    {
        RCLCPP_INFO(this->get_logger(),
            "Voltage: %.2f V | Percentage: %.1f%% | %.2f Minutes of Operation Remaining",
            voltage, percent, time_est);
    }
}

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<BatteryReader>());
    rclcpp::shutdown();
    return 0;
}
