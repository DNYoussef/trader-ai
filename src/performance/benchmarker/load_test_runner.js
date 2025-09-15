/**
 * Real Load Test Runner
 * Executes actual load tests against real systems with measurable performance metrics
 */

const http = require('http');
const https = require('https');
const cluster = require('cluster');
const os = require('os');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

class RealLoadTestRunner {
    constructor(config = {}) {
        this.config = {
            targetUrl: config.targetUrl || 'http://localhost:3000',
            concurrency: config.concurrency || 10,
            duration: config.duration || 60000, // 60 seconds
            rampUpTime: config.rampUpTime || 5000, // 5 seconds
            requestsPerSecond: config.requestsPerSecond || 100,
            timeout: config.timeout || 5000,
            ...config
        };

        this.results = {
            totalRequests: 0,
            successfulRequests: 0,
            failedRequests: 0,
            totalResponseTime: 0,
            minResponseTime: Infinity,
            maxResponseTime: 0,
            responseTimes: [],
            errors: [],
            startTime: null,
            endTime: null,
            memoryUsage: [],
            cpuUsage: []
        };

        this.workers = new Map();
        this.isRunning = false;
        this.startMemory = process.memoryUsage();
        this.startCpu = process.cpuUsage();
    }

    /**
     * Execute real load test with actual HTTP requests
     */
    async executeLoadTest() {
        console.log('[LOAD-TEST] Starting real load test execution...');
        console.log(`Target: ${this.config.targetUrl}`);
        console.log(`Concurrency: ${this.config.concurrency}`);
        console.log(`Duration: ${this.config.duration}ms`);
        console.log(`Expected RPS: ${this.config.requestsPerSecond}`);

        this.results.startTime = Date.now();
        this.isRunning = true;

        try {
            // Start resource monitoring
            const monitoringInterval = this.startResourceMonitoring();

            // Execute load test
            if (this.config.concurrency > 1) {
                await this.executeClusteredLoadTest();
            } else {
                await this.executeSingleThreadLoadTest();
            }

            // Stop monitoring
            clearInterval(monitoringInterval);

            this.results.endTime = Date.now();
            this.isRunning = false;

            return this.generateLoadTestReport();

        } catch (error) {
            this.isRunning = false;
            console.error('[LOAD-TEST] Load test failed:', error);
            throw error;
        }
    }

    /**
     * Execute load test using multiple worker processes
     */
    async executeClusteredLoadTest() {
        return new Promise((resolve, reject) => {
            if (cluster.isMaster) {
                console.log(`[LOAD-TEST] Master process starting ${this.config.concurrency} workers...`);

                const workerResults = [];
                let completedWorkers = 0;

                // Fork worker processes
                for (let i = 0; i < this.config.concurrency; i++) {
                    const worker = cluster.fork({
                        WORKER_ID: i,
                        TARGET_URL: this.config.targetUrl,
                        DURATION: this.config.duration,
                        RPS_PER_WORKER: Math.floor(this.config.requestsPerSecond / this.config.concurrency),
                        TIMEOUT: this.config.timeout
                    });

                    this.workers.set(worker.id, {
                        worker,
                        startTime: Date.now(),
                        requests: 0,
                        success: 0,
                        errors: 0
                    });

                    worker.on('message', (message) => {
                        if (message.type === 'result') {
                            workerResults.push(message.data);
                            completedWorkers++;

                            if (completedWorkers === this.config.concurrency) {
                                this.aggregateWorkerResults(workerResults);
                                resolve();
                            }
                        }
                    });

                    worker.on('error', (error) => {
                        console.error(`[LOAD-TEST] Worker ${worker.id} error:`, error);
                        reject(error);
                    });
                }

                // Cleanup workers after test
                setTimeout(() => {
                    this.workers.forEach(({ worker }) => {
                        worker.kill();
                    });
                }, this.config.duration + 10000);

            } else {
                // Worker process
                this.runWorkerLoadTest();
            }
        });
    }

    /**
     * Run load test in worker process
     */
    async runWorkerLoadTest() {
        const workerId = process.env.WORKER_ID;
        const targetUrl = process.env.TARGET_URL;
        const duration = parseInt(process.env.DURATION);
        const rpsPerWorker = parseInt(process.env.RPS_PER_WORKER);
        const timeout = parseInt(process.env.TIMEOUT);

        console.log(`[WORKER-${workerId}] Starting load test worker`);

        const workerResults = {
            workerId,
            requests: 0,
            successful: 0,
            failed: 0,
            responseTimes: [],
            errors: [],
            startTime: Date.now()
        };

        const startTime = Date.now();
        const intervalMs = 1000 / rpsPerWorker;

        const makeRequest = () => {
            if (Date.now() - startTime >= duration) {
                // Send results back to master
                process.send({
                    type: 'result',
                    data: {
                        ...workerResults,
                        endTime: Date.now()
                    }
                });
                return;
            }

            const requestStart = Date.now();
            workerResults.requests++;

            this.performHttpRequest(targetUrl, timeout)
                .then((responseTime) => {
                    workerResults.successful++;
                    workerResults.responseTimes.push(responseTime);
                })
                .catch((error) => {
                    workerResults.failed++;
                    workerResults.errors.push({
                        timestamp: Date.now(),
                        error: error.message
                    });
                })
                .finally(() => {
                    // Schedule next request
                    setTimeout(makeRequest, intervalMs);
                });
        };

        // Start making requests
        makeRequest();
    }

    /**
     * Execute single-threaded load test
     */
    async executeSingleThreadLoadTest() {
        console.log('[LOAD-TEST] Executing single-threaded load test...');

        const startTime = Date.now();
        const intervalMs = 1000 / this.config.requestsPerSecond;
        let requestCount = 0;

        return new Promise((resolve) => {
            const makeRequest = () => {
                if (Date.now() - startTime >= this.config.duration) {
                    resolve();
                    return;
                }

                requestCount++;
                this.results.totalRequests++;

                const requestStart = Date.now();

                this.performHttpRequest(this.config.targetUrl, this.config.timeout)
                    .then((responseTime) => {
                        this.results.successfulRequests++;
                        this.results.totalResponseTime += responseTime;
                        this.results.responseTimes.push(responseTime);
                        this.results.minResponseTime = Math.min(this.results.minResponseTime, responseTime);
                        this.results.maxResponseTime = Math.max(this.results.maxResponseTime, responseTime);
                    })
                    .catch((error) => {
                        this.results.failedRequests++;
                        this.results.errors.push({
                            timestamp: Date.now(),
                            error: error.message
                        });
                    })
                    .finally(() => {
                        // Schedule next request
                        setTimeout(makeRequest, intervalMs);
                    });
            };

            // Start making requests
            makeRequest();
        });
    }

    /**
     * Perform actual HTTP request with real measurement
     */
    performHttpRequest(url, timeout) {
        return new Promise((resolve, reject) => {
            const startTime = Date.now();
            const isHttps = url.startsWith('https');
            const client = isHttps ? https : http;

            // Parse URL
            const urlObj = new URL(url);
            const options = {
                hostname: urlObj.hostname,
                port: urlObj.port || (isHttps ? 443 : 80),
                path: urlObj.pathname + urlObj.search,
                method: 'GET',
                timeout: timeout,
                headers: {
                    'User-Agent': 'RealLoadTestRunner/1.0.0',
                    'Accept': 'application/json,text/html,*/*',
                    'Connection': 'keep-alive'
                }
            };

            const req = client.request(options, (res) => {
                let data = '';

                res.on('data', (chunk) => {
                    data += chunk;
                });

                res.on('end', () => {
                    const responseTime = Date.now() - startTime;

                    if (res.statusCode >= 200 && res.statusCode < 400) {
                        resolve(responseTime);
                    } else {
                        reject(new Error(`HTTP ${res.statusCode}: ${res.statusMessage}`));
                    }
                });
            });

            req.on('error', (error) => {
                reject(error);
            });

            req.on('timeout', () => {
                req.destroy();
                reject(new Error('Request timeout'));
            });

            req.end();
        });
    }

    /**
     * Aggregate results from worker processes
     */
    aggregateWorkerResults(workerResults) {
        console.log('[LOAD-TEST] Aggregating results from workers...');

        this.results.totalRequests = workerResults.reduce((sum, worker) => sum + worker.requests, 0);
        this.results.successfulRequests = workerResults.reduce((sum, worker) => sum + worker.successful, 0);
        this.results.failedRequests = workerResults.reduce((sum, worker) => sum + worker.failed, 0);

        // Combine all response times
        this.results.responseTimes = workerResults.flatMap(worker => worker.responseTimes || []);
        this.results.totalResponseTime = this.results.responseTimes.reduce((sum, time) => sum + time, 0);

        // Calculate min/max response times
        if (this.results.responseTimes.length > 0) {
            this.results.minResponseTime = Math.min(...this.results.responseTimes);
            this.results.maxResponseTime = Math.max(...this.results.responseTimes);
        }

        // Combine errors
        this.results.errors = workerResults.flatMap(worker => worker.errors || []);
    }

    /**
     * Start monitoring system resources during load test
     */
    startResourceMonitoring() {
        console.log('[MONITOR] Starting resource monitoring...');

        return setInterval(() => {
            const currentMemory = process.memoryUsage();
            const currentCpu = process.cpuUsage();

            this.results.memoryUsage.push({
                timestamp: Date.now(),
                rss: currentMemory.rss,
                heapUsed: currentMemory.heapUsed,
                heapTotal: currentMemory.heapTotal,
                external: currentMemory.external
            });

            this.results.cpuUsage.push({
                timestamp: Date.now(),
                user: currentCpu.user,
                system: currentCpu.system
            });
        }, 1000);
    }

    /**
     * Generate comprehensive load test report
     */
    generateLoadTestReport() {
        const duration = this.results.endTime - this.results.startTime;
        const successRate = (this.results.successfulRequests / this.results.totalRequests) * 100;
        const actualRps = (this.results.totalRequests / duration) * 1000;
        const averageResponseTime = this.results.totalResponseTime / this.results.successfulRequests || 0;

        // Calculate percentiles
        const sortedResponseTimes = [...this.results.responseTimes].sort((a, b) => a - b);
        const p50 = this.calculatePercentile(sortedResponseTimes, 50);
        const p95 = this.calculatePercentile(sortedResponseTimes, 95);
        const p99 = this.calculatePercentile(sortedResponseTimes, 99);

        // Calculate resource overhead
        const endMemory = process.memoryUsage();
        const endCpu = process.cpuUsage();
        const memoryOverhead = ((endMemory.rss - this.startMemory.rss) / this.startMemory.rss) * 100;
        const cpuUsage = ((endCpu.user + endCpu.system) - (this.startCpu.user + this.startCpu.system)) / 1000000;

        const report = {
            // Test Configuration
            configuration: {
                targetUrl: this.config.targetUrl,
                concurrency: this.config.concurrency,
                expectedRps: this.config.requestsPerSecond,
                duration: this.config.duration,
                timeout: this.config.timeout
            },

            // Performance Metrics
            performance: {
                totalRequests: this.results.totalRequests,
                successfulRequests: this.results.successfulRequests,
                failedRequests: this.results.failedRequests,
                successRate: parseFloat(successRate.toFixed(2)),
                actualRps: parseFloat(actualRps.toFixed(2)),
                rpsVariance: parseFloat(((actualRps - this.config.requestsPerSecond) / this.config.requestsPerSecond * 100).toFixed(2))
            },

            // Latency Metrics (all in milliseconds)
            latency: {
                average: parseFloat(averageResponseTime.toFixed(2)),
                min: this.results.minResponseTime,
                max: this.results.maxResponseTime,
                p50: p50,
                p95: p95,
                p99: p99
            },

            // Resource Usage
            resources: {
                memoryOverheadPercent: parseFloat(memoryOverhead.toFixed(2)),
                cpuUsageMilliseconds: parseFloat(cpuUsage.toFixed(2)),
                peakMemoryMB: Math.max(...this.results.memoryUsage.map(m => m.rss)) / 1024 / 1024,
                averageMemoryMB: (this.results.memoryUsage.reduce((sum, m) => sum + m.rss, 0) / this.results.memoryUsage.length) / 1024 / 1024
            },

            // Compliance
            compliance: {
                overheadCompliant: memoryOverhead <= 2.0, // <2% overhead requirement
                latencyCompliant: p95 <= 1000, // <1000ms P95 requirement
                throughputCompliant: actualRps >= this.config.requestsPerSecond * 0.9, // 90% of expected RPS
                successRateCompliant: successRate >= 95 // 95% success rate requirement
            },

            // Errors and Issues
            errors: {
                totalErrors: this.results.errors.length,
                errorRate: (this.results.failedRequests / this.results.totalRequests) * 100,
                errorTypes: this.categorizeErrors()
            },

            // Test Execution
            execution: {
                startTime: new Date(this.results.startTime).toISOString(),
                endTime: new Date(this.results.endTime).toISOString(),
                actualDuration: duration,
                executionEnvironment: {
                    nodeVersion: process.version,
                    platform: process.platform,
                    cpuArchitecture: process.arch,
                    totalMemory: os.totalmem(),
                    freeMemory: os.freemem(),
                    cpuCores: os.cpus().length
                }
            }
        };

        // Add overall assessment
        report.assessment = this.assessLoadTestResults(report);

        return report;
    }

    /**
     * Calculate percentile value from sorted array
     */
    calculatePercentile(sortedArray, percentile) {
        if (sortedArray.length === 0) return 0;

        const index = (percentile / 100) * (sortedArray.length - 1);
        const lower = Math.floor(index);
        const upper = Math.ceil(index);

        if (lower === upper) {
            return sortedArray[lower];
        }

        return sortedArray[lower] + (sortedArray[upper] - sortedArray[lower]) * (index - lower);
    }

    /**
     * Categorize errors by type
     */
    categorizeErrors() {
        const errorTypes = {};

        this.results.errors.forEach(error => {
            const errorType = this.categorizeError(error.error);
            errorTypes[errorType] = (errorTypes[errorType] || 0) + 1;
        });

        return errorTypes;
    }

    /**
     * Categorize individual error
     */
    categorizeError(errorMessage) {
        if (errorMessage.includes('timeout')) return 'timeout';
        if (errorMessage.includes('ECONNREFUSED')) return 'connection_refused';
        if (errorMessage.includes('ENOTFOUND')) return 'dns_resolution';
        if (errorMessage.includes('HTTP 4')) return 'client_error';
        if (errorMessage.includes('HTTP 5')) return 'server_error';
        return 'unknown';
    }

    /**
     * Assess overall load test results
     */
    assessLoadTestResults(report) {
        const compliance = report.compliance;
        const passedChecks = Object.values(compliance).filter(check => check).length;
        const totalChecks = Object.keys(compliance).length;
        const complianceScore = (passedChecks / totalChecks) * 100;

        let grade, recommendation;

        if (complianceScore >= 100) {
            grade = 'A+';
            recommendation = 'EXCELLENT - System exceeds all performance requirements';
        } else if (complianceScore >= 75) {
            grade = 'A';
            recommendation = 'GOOD - System meets most performance requirements';
        } else if (complianceScore >= 50) {
            grade = 'B';
            recommendation = 'ACCEPTABLE - Some performance issues need attention';
        } else {
            grade = 'C';
            recommendation = 'NEEDS IMPROVEMENT - Significant performance issues detected';
        }

        return {
            grade,
            complianceScore: parseFloat(complianceScore.toFixed(1)),
            passedChecks,
            totalChecks,
            recommendation,
            productionReady: complianceScore >= 75
        };
    }

    /**
     * Save results to file
     */
    async saveResults(report, outputPath = './load-test-results.json') {
        const outputDir = path.dirname(outputPath);
        if (!fs.existsSync(outputDir)) {
            fs.mkdirSync(outputDir, { recursive: true });
        }

        fs.writeFileSync(outputPath, JSON.stringify(report, null, 2));
        console.log(`[SAVE] Load test results saved to: ${outputPath}`);

        return outputPath;
    }
}

// Execute load test if run directly
if (require.main === module) {
    const config = {
        targetUrl: process.env.TARGET_URL || 'http://localhost:3000',
        concurrency: parseInt(process.env.CONCURRENCY) || 5,
        duration: parseInt(process.env.DURATION) || 30000,
        requestsPerSecond: parseInt(process.env.RPS) || 50,
        timeout: parseInt(process.env.TIMEOUT) || 5000
    };

    console.log('[LOAD-TEST] Starting real load test execution...');
    console.log('Configuration:', config);

    const loadTest = new RealLoadTestRunner(config);

    loadTest.executeLoadTest()
        .then(async (report) => {
            console.log('\n[RESULTS] Load Test Completed Successfully!');
            console.log('='.repeat(60));
            console.log(`Performance Grade: ${report.assessment.grade}`);
            console.log(`Success Rate: ${report.performance.successRate}%`);
            console.log(`Actual RPS: ${report.performance.actualRps}`);
            console.log(`P95 Latency: ${report.latency.p95}ms`);
            console.log(`Memory Overhead: ${report.resources.memoryOverheadPercent}%`);
            console.log(`Production Ready: ${report.assessment.productionReady ? 'YES' : 'NO'}`);
            console.log('='.repeat(60));

            // Save detailed results
            await loadTest.saveResults(report, './tests/performance/load-test-results.json');

            process.exit(report.assessment.productionReady ? 0 : 1);
        })
        .catch((error) => {
            console.error('[ERROR] Load test failed:', error);
            process.exit(1);
        });
}

module.exports = RealLoadTestRunner;