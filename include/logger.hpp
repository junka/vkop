// Copyright 2025 @junka
#ifndef LOGGER_H
#define LOGGER_H

#include <cstring>
#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <mutex>
#include <ctime>

enum LogLevel {
    LOG_INFO,
    LOG_WARNING,
    LOG_ERROR,
};

class Logger {
public:
    static Logger& getInstance() {
        static Logger instance;
        return instance;
    }

    void setLevel(LogLevel level) {
        logLevel_ = level;
    }

    void enableFileOutput(const std::string& filePath, bool enable = true) {
        fileOutputEnabled_ = enable;
        if (enable) {
            logFile_.open(filePath, std::ofstream::out | std::ofstream::app);
            if (!logFile_.is_open()) {
                std::cerr << "Failed to open log file: " << filePath << std::endl;
            }
        } else {
            if (logFile_.is_open()) {
                logFile_.close();
            }
        }
    }

    template <typename... Args>
    void info(const std::string& format, Args... args) {
        log(LOG_INFO, "[INFO] " + format, args...);
    }

    template <typename... Args>
    void warning(const std::string& format, Args... args) {
        log(LOG_WARNING, "[WARNING] " + format, args...);
    }

    template <typename... Args>
    void error(const std::string& format, Args... args) {
        log(LOG_ERROR, "[ERROR] " + format, args...);
    }

    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
private:
    LogLevel logLevel_ = LOG_INFO;
    bool fileOutputEnabled_ = false;
    std::ofstream logFile_;
    std::mutex mutex_;

    Logger() = default;
    ~Logger() {
        if (logFile_.is_open()) {
            logFile_.close();
        }
    }

    template <typename... Args>
    void log(LogLevel level, const std::string& format, Args... args) {
        if (level < logLevel_) return;

        std::lock_guard<std::mutex> lock(mutex_);
        auto now = std::chrono::system_clock::now();
        auto time_now = std::chrono::system_clock::to_time_t(now);
        char time_str[25];
        std::strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", std::localtime(&time_now));

        // Format the message
        size_t size;
        std::unique_ptr<char[]> buffer;
        if constexpr (sizeof...(args) == 0) {
            size = format.size() + 1;
            buffer = std::make_unique<char[]>(size);
            strncpy(buffer.get(), format.c_str(), size);
        } else {
            size = snprintf(nullptr, 0, format.c_str(), args...) + 1;
            buffer = std::make_unique<char[]>(size);
            snprintf(buffer.get(), size, format.c_str(), args...);
        }
        std::string formatted_message(buffer.get(), buffer.get() + size - 1);

        std::cout << time_str << " " << formatted_message << std::endl;
        if (fileOutputEnabled_ && logFile_.is_open()) {
            logFile_ << time_str << " " << formatted_message << std::endl;
        }
    }
};


#define LOG_INFO(...) Logger::getInstance().info(__VA_ARGS__)
#define LOG_ERROR(...) Logger::getInstance().error(__VA_ARGS__)
#define LOG_WARN(...) Logger::getInstance().warning(__VA_ARGS__)

#endif // LOGGER_H
