#ifndef HNSW_LOGGING_H
#define HNSW_LOGGING_H

#include <iostream>
#include <fstream>
#include <string>
#include <mutex>
#include <memory>
#include <atomic>

namespace hnswlib {

// Logging modes
enum class LogMode {
    RUN,    // No logging
    DEBUG   // Full logging
};

// Logging levels
enum class LogLevel {
    ERROR,
    WARN,
    INFO,
    DEBUG,
    TRACE
};

class Logger {
private:
    static std::unique_ptr<Logger> instance_;
    static std::mutex instance_mutex_;
    
    LogMode mode_;
    LogLevel level_;
    std::ofstream log_file_;
    std::ostream* output_stream_;
    std::mutex log_mutex_;
    std::string log_file_path_;
    /** When false, no log lines are emitted (cheap check in \c isEnabled). Independent of \c LogMode. */
    std::atomic<bool> output_enabled_{true};

    Logger() : mode_(LogMode::RUN), level_(LogLevel::INFO), output_stream_(&std::cerr) {}
    
public:
    static Logger& getInstance() {
        std::lock_guard<std::mutex> lock(instance_mutex_);
        if (!instance_) {
            instance_ = std::unique_ptr<Logger>(new Logger());
        }
        return *instance_;
    }
    
    // Initialize logger with mode and optional log file
    // Note: Log file is opened in APPEND mode (std::ios::app) to preserve previous logs
    void init(LogMode mode, LogLevel level = LogLevel::INFO, const std::string& log_file_path = "") {
        std::lock_guard<std::mutex> lock(log_mutex_);
        mode_ = mode;
        level_ = level;
        
        if (mode_ == LogMode::DEBUG && !log_file_path.empty()) {
            log_file_path_ = log_file_path;
            
            // Close existing log file if open
            if (log_file_.is_open()) {
                log_file_.close();
            }
            
            // Open in append mode to preserve previous logs
            log_file_.open(log_file_path_, std::ios::out | std::ios::app);
            
            if (log_file_.is_open()) {
                output_stream_ = &log_file_;
                // Write separator for new session
                *output_stream_ << "\n========================================\n";
                *output_stream_ << "New logging session started\n";
                *output_stream_ << "========================================\n";
                output_stream_->flush();
            } else {
                std::cerr << "[ERROR] Failed to open log file: " << log_file_path_ << std::endl;
                output_stream_ = &std::cerr;
            }
        } else {
            output_stream_ = &std::cerr;
        }
    }

    /** Master switch: disable to suppress all \c HNSW_LOG_* output at runtime (e.g. benchmarks). */
    void setOutputEnabled(bool enabled) {
        output_enabled_.store(enabled, std::memory_order_relaxed);
    }

    bool isOutputEnabled() const {
        return output_enabled_.load(std::memory_order_relaxed);
    }
    
    // Check if logging is enabled for given level
    bool isEnabled(LogLevel level) const {
        return output_enabled_.load(std::memory_order_relaxed) && mode_ == LogMode::DEBUG && level <= level_;
    }
    
    // Log message with level
    template<typename... Args>
    void log(LogLevel level, Args&&... args) {
        if (!isEnabled(level)) return;
        
        std::lock_guard<std::mutex> lock(log_mutex_);
        (void)((*output_stream_ << ... << std::forward<Args>(args)) << std::endl);
        output_stream_->flush();
    }
    
    // Convenience methods for different log levels
    template<typename... Args>
    void error(Args&&... args) {
        log(LogLevel::ERROR, "[ERROR] ", std::forward<Args>(args)...);
    }
    
    template<typename... Args>
    void warn(Args&&... args) {
        log(LogLevel::WARN, "[WARN] ", std::forward<Args>(args)...);
    }
    
    template<typename... Args>
    void info(Args&&... args) {
        log(LogLevel::INFO, "[INFO] ", std::forward<Args>(args)...);
    }
    
    template<typename... Args>
    void debug(Args&&... args) {
        log(LogLevel::DEBUG, "[DEBUG] ", std::forward<Args>(args)...);
    }
    
    template<typename... Args>
    void trace(Args&&... args) {
        log(LogLevel::TRACE, "[TRACE] ", std::forward<Args>(args)...);
    }
    
    // Close log file
    void close() {
        std::lock_guard<std::mutex> lock(log_mutex_);
        if (log_file_.is_open()) {
            log_file_.close();
        }
    }
    
    ~Logger() {
        close();
    }
};

// Global logger instance
extern Logger& logger;

// Convenience macros for logging
#define HNSW_LOG_ERROR(...) hnswlib::logger.error(__VA_ARGS__)
#define HNSW_LOG_WARN(...)  hnswlib::logger.warn(__VA_ARGS__)
#define HNSW_LOG_INFO(...)  hnswlib::logger.info(__VA_ARGS__)
#define HNSW_LOG_DEBUG(...) hnswlib::logger.debug(__VA_ARGS__)
#define HNSW_LOG_TRACE(...) hnswlib::logger.trace(__VA_ARGS__)

// Conditional logging macros
#define HNSW_LOG_IF(condition, level, ...) \
    if (condition) { \
        hnswlib::logger.log(level, __VA_ARGS__); \
    }

// Initialize logging system
inline void initLogging(LogMode mode, LogLevel level = LogLevel::INFO, const std::string& log_file_path = "") {
    logger.init(mode, level, log_file_path);
}

// Set logging mode
inline void setLogMode(LogMode mode) {
    logger.init(mode, logger.isEnabled(LogLevel::DEBUG) ? LogLevel::DEBUG : LogLevel::INFO);
}

// Check if debug mode is enabled
inline bool isDebugMode() {
    return logger.isEnabled(LogLevel::DEBUG);
}

/** Global runtime toggle for \c HNSW_LOG_* (used by \c hnswalg.h and related code). */
inline void setHnswLoggingOutputEnabled(bool enabled) {
    Logger::getInstance().setOutputEnabled(enabled);
}

inline bool isHnswLoggingOutputEnabled() {
    return Logger::getInstance().isOutputEnabled();
}

} // namespace hnswlib

#endif // HNSW_LOGGING_H 