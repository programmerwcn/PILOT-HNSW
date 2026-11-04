#include "logging.h"

namespace hnswlib {

// Initialize static members
std::unique_ptr<Logger> Logger::instance_ = nullptr;
std::mutex Logger::instance_mutex_;

// Global logger instance
Logger& logger = Logger::getInstance();

} // namespace hnswlib 