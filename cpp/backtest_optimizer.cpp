#include <vector>
#include <map>
#include <thread>
#include <mutex>
#include <queue>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <limits>
#include <cstring>

// ============================================================================
// Data Structures
// ============================================================================

struct PriceData {
    std::vector<double> close_prices;
    size_t num_records;
};

struct BacktestResult {
    int short_window;
    int long_window;
    double sharpe_ratio;
    double final_value;
    double max_drawdown;
};

struct WorkItem {
    int short_window;
    int long_window;
};

// ============================================================================
// Global State (thread-safe)
// ============================================================================

std::mutex result_mutex;
BacktestResult best_result = {0, 0, std::numeric_limits<double>::lowest(), 0.0, 0.0};
int completed_tests = 0;
int total_tests = 0;

// ============================================================================
// SMA Calculation (mirroring Python logic)
// ============================================================================

std::vector<double> calculate_sma(const std::vector<double>& prices, int window) {
    std::vector<double> sma;
    
    if (window < 1 || window > (int)prices.size()) {
        return sma; // Return empty on error
    }
    
    // First (window - 1) values are None/NaN
    for (int i = 0; i < window - 1; ++i) {
        sma.push_back(std::numeric_limits<double>::quiet_NaN());
    }
    
    // Calculate first SMA value
    double rolling_sum = 0.0;
    for (int i = 0; i < window; ++i) {
        rolling_sum += prices[i];
    }
    sma.push_back(rolling_sum / window);
    
    // Rolling window calculation
    for (int i = window; i < (int)prices.size(); ++i) {
        rolling_sum += prices[i] - prices[i - window];
        sma.push_back(rolling_sum / window);
    }
    
    return sma;
}

// ============================================================================
// Signal Generation (mirroring Python strategy logic)
// ============================================================================

struct Signal {
    int index;
    double close_price;
    int signal; // 1 = BUY, -1 = SELL, 0 = HOLD
};

std::vector<Signal> generate_signals(
    const std::vector<double>& prices,
    const std::vector<double>& short_sma,
    const std::vector<double>& long_sma
) {
    std::vector<Signal> signals;
    
    for (size_t i = 0; i < prices.size(); ++i) {
        double s_curr = short_sma[i];
        double l_curr = long_sma[i];
        
        // Skip if either SMA is NaN
        if (std::isnan(s_curr) || std::isnan(l_curr)) {
            signals.push_back({(int)i, prices[i], 0});
            continue;
        }
        
        if (i == 0) {
            signals.push_back({(int)i, prices[i], 0});
            continue;
        }
        
        double s_prev = short_sma[i - 1];
        double l_prev = long_sma[i - 1];
        
        // Skip if previous SMA was NaN
        if (std::isnan(s_prev) || std::isnan(l_prev)) {
            signals.push_back({(int)i, prices[i], 0});
            continue;
        }
        
        // Golden Cross: fast crosses above slow (BUY)
        if (s_prev <= l_prev && s_curr > l_curr) {
            signals.push_back({(int)i, prices[i], 1});
        }
        // Death Cross: fast crosses below slow (SELL)
        else if (s_prev >= l_prev && s_curr < l_curr) {
            signals.push_back({(int)i, prices[i], -1});
        }
        // No signal
        else {
            signals.push_back({(int)i, prices[i], 0});
        }
    }
    
    return signals;
}

// ============================================================================
// Portfolio Simulation
// ============================================================================

std::vector<double> simulate_portfolio(
    const std::vector<Signal>& signals,
    double initial_cash,
    double commission
) {
    std::vector<double> portfolio_values;
    double cash = initial_cash;
    double shares = 0.0;
    
    for (const auto& signal : signals) {
        if (signal.signal == 1) { // BUY
            int quantity = (int)(cash / (signal.close_price * (1.0 + commission)));
            if (quantity > 0) {
                double cost = quantity * signal.close_price * (1.0 + commission);
                cash -= cost;
                shares += quantity;
            }
        } else if (signal.signal == -1 && shares > 0) { // SELL
            double proceeds = shares * signal.close_price * (1.0 - commission);
            cash += proceeds;
            shares = 0.0;
        }
        
        // Calculate current portfolio value
        double portfolio_value = cash + (shares * signal.close_price);
        portfolio_values.push_back(portfolio_value);
    }
    
    return portfolio_values;
}

// ============================================================================
// Metrics Calculation
// ============================================================================

double calculate_max_drawdown(const std::vector<double>& portfolio_values) {
    if (portfolio_values.empty()) {
        return 0.0;
    }
    
    double peak = portfolio_values[0];
    double max_dd = 0.0;
    
    for (double value : portfolio_values) {
        if (value > peak) {
            peak = value;
        }
        double dd = (value - peak) / peak;
        if (dd < max_dd) {
            max_dd = dd;
        }
    }
    
    return max_dd;
}

double calculate_sharpe_ratio(const std::vector<double>& portfolio_values) {
    if (portfolio_values.size() < 2) {
        return 0.0;
    }
    
    const int periods_per_year = 252;
    const double risk_free_rate = 0.0;
    const double daily_rate = risk_free_rate / periods_per_year;
    
    // Calculate daily returns
    std::vector<double> excess_returns;
    for (size_t i = 1; i < portfolio_values.size(); ++i) {
        double daily_return = (portfolio_values[i] - portfolio_values[i - 1]) / portfolio_values[i - 1];
        double excess_return = daily_return - daily_rate;
        excess_returns.push_back(excess_return);
    }
    
    if (excess_returns.empty()) {
        return 0.0;
    }
    
    // Calculate mean
    double mean = std::accumulate(excess_returns.begin(), excess_returns.end(), 0.0) / excess_returns.size();
    
    // Calculate standard deviation
    double variance = 0.0;
    for (double ret : excess_returns) {
        variance += (ret - mean) * (ret - mean);
    }
    variance /= excess_returns.size();
    double std_dev = std::sqrt(variance);
    
    if (std_dev == 0.0) {
        return 0.0;
    }
    
    return (mean / std_dev) * std::sqrt(periods_per_year);
}

// ============================================================================
// Single Backtest Execution
// ============================================================================

BacktestResult run_single_backtest(
    const PriceData& data,
    int short_window,
    int long_window,
    double initial_cash,
    double commission
) {
    // Validate parameters
    if (short_window >= long_window || short_window < 1) {
        return {short_window, long_window, std::numeric_limits<double>::lowest(), 0.0, 0.0};
    }
    
    // Calculate SMAs
    auto short_sma = calculate_sma(data.close_prices, short_window);
    auto long_sma = calculate_sma(data.close_prices, long_window);
    
    if (short_sma.empty() || long_sma.empty()) {
        return {short_window, long_window, std::numeric_limits<double>::lowest(), 0.0, 0.0};
    }
    
    // Generate signals
    auto signals = generate_signals(data.close_prices, short_sma, long_sma);
    
    // Simulate portfolio
    auto portfolio_values = simulate_portfolio(signals, initial_cash, commission);
    
    if (portfolio_values.empty()) {
        return {short_window, long_window, std::numeric_limits<double>::lowest(), 0.0, 0.0};
    }
    
    // Calculate metrics
    double sharpe = calculate_sharpe_ratio(portfolio_values);
    double mdd = calculate_max_drawdown(portfolio_values);
    double final_value = portfolio_values.back();
    
    return {short_window, long_window, sharpe, final_value, mdd};
}

// ============================================================================
// Worker Thread Function
// ============================================================================

void worker_thread(
    const PriceData& data,
    std::queue<WorkItem>& work_queue,
    std::mutex& queue_mutex,
    double initial_cash,
    double commission
) {
    while (true) {
        WorkItem item;
        
        // Get next work item
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            if (work_queue.empty()) {
                break;
            }
            item = work_queue.front();
            work_queue.pop();
        }
        
        // Run backtest
        auto result = run_single_backtest(data, item.short_window, item.long_window, initial_cash, commission);
        
        // Update best result (thread-safe)
        {
            std::lock_guard<std::mutex> lock(result_mutex);
            completed_tests++;
            if (result.sharpe_ratio > best_result.sharpe_ratio) {
                best_result = result;
            }
        }
    }
}

// ============================================================================
// Main Grid Search Function (C interface)
// ============================================================================

extern "C" {
    
    // Structure returned to Python
    typedef struct {
        int short_window;
        int long_window;
        double sharpe_ratio;
        double final_value;
        double max_drawdown;
    } OptimizationResult;
    
    /**
     * Run grid search with multithreading
     * 
     * Args:
     *   prices: Array of closing prices
     *   num_prices: Length of prices array
     *   fast_min, fast_max: Range for short SMA (inclusive/exclusive)
     *   slow_min, slow_max: Range for long SMA (inclusive/exclusive)
     *   initial_cash: Starting capital
     *   commission: Transaction commission as decimal
     *   num_threads: Number of worker threads
     *   result: Pointer to OptimizationResult struct to fill
     * 
     * Returns: 0 on success, non-zero on error
     */
    int grid_search_multithreaded(
        const double* prices,
        int num_prices,
        int fast_min,
        int fast_max,
        int slow_min,
        int slow_max,
        double initial_cash,
        double commission,
        int num_threads,
        OptimizationResult* result
    ) {
        // Input validation
        if (!prices || num_prices <= 0 || !result) {
            return 1;
        }
        
        if (fast_min < 1 || fast_max <= fast_min || slow_min < 1 || slow_max <= slow_min) {
            return 2;
        }
        
        // Prepare data
        PriceData data;
        data.close_prices.assign(prices, prices + num_prices);
        data.num_records = num_prices;
        
        // Create work queue
        std::queue<WorkItem> work_queue;
        int total_combinations = 0;
        
        for (int fast = fast_min; fast < fast_max; ++fast) {
            for (int slow = slow_min; slow < slow_max; ++slow) {
                if (fast < slow) {
                    work_queue.push({fast, slow});
                    total_combinations++;
                }
            }
        }
        
        if (total_combinations == 0) {
            return 3;
        }
        
        // Initialize global state
        best_result = {0, 0, std::numeric_limits<double>::lowest(), 0.0, 0.0};
        completed_tests = 0;
        total_tests = total_combinations;
        
        // Clamp num_threads
        if (num_threads < 1) {
            num_threads = std::thread::hardware_concurrency();
        }
        if (num_threads > total_combinations) {
            num_threads = total_combinations;
        }
        
        // Create and launch worker threads
        std::vector<std::thread> threads;
        std::mutex queue_mutex;
        
        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back(
                worker_thread,
                std::ref(data),
                std::ref(work_queue),
                std::ref(queue_mutex),
                initial_cash,
                commission
            );
        }
        
        // Wait for all threads to complete
        for (auto& t : threads) {
            t.join();
        }
        
        // Return result
        result->short_window = best_result.short_window;
        result->long_window = best_result.long_window;
        result->sharpe_ratio = best_result.sharpe_ratio;
        result->final_value = best_result.final_value;
        result->max_drawdown = best_result.max_drawdown;
        
        return 0; // Success
    }
    
    /**
     * Free any allocated resources (currently a no-op)
     */
    void cleanup() {
        // No cleanup needed currently
    }
}
