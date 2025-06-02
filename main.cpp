#include <cmath>
#include <cstring>
#include <cfloat>
#include "esp_dsp.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "dspm_matrix.h"
#include "esp_timer.h"
#include <stdio.h>
#include <iostream>
#include "esp_heap_caps.h"
#include "esp_cpu.h"
#include <map>
#include <string>
#include <sstream>
constexpr int NUM_JOINTS = 6;
/*constexpr float ROT_TRANS[2][NUM_JOINTS][3] = {{{0, 0, 0}, {1.5708, -0.10095, -3.1416}, {0, 0, -1.759},
                                                 {1.5708, 0, 0}, {-1.5708, 0, 0}, {-1.5708, 0, 0}},
                                                {{0, 0, 0.123}, {0, 0, 0}, {0.28503, 0, 0},
                                                {-0.021984, -0.25075, 0}, {0, 0, 0}, {0, -0.091, 0}}};
*/

constexpr float ROT_TRANS[2][NUM_JOINTS][3] = {
    // Rotation components (theta_x, theta_y, theta_z) for each joint
    {
        {0, 0, 0},          // Joint 1
        {1.5708, -0.10095, -3.1416},  // Joint 2
        {0, 0, -1.759},     // Joint 3
        {1.5708, 0, 0},     // Joint 4
        {-1.5708, 0, 0},    // Joint 5
        {-1.5708, 0, 0}     // Joint 6
    },
    // Translation components (x, y, z) for each joint
    {
        {0, 0, 0.123},      // Joint 1
        {0, 0, 0},          // Joint 2
        {0.28503, 0, 0},    // Joint 3
        {-0.021984, -0.25075, 0},  // Joint 4
        {0, 0, 0},          // Joint 5
        {0, -0.091, 0}      // Joint 6
    }
};

struct IKConfig {
    int max_iterations = 1000;
    float tolerance = 0.01f;
    float nr_step_size = 0.1f;      // For Newton-Raphson
    float dls_step_size = 0.1f;     // For DLS
    float dls_damping = 0.01f;      // Damping factor for DLS
    float ccd_epsilon = 1e-6f;      // Small value to avoid division by zero
    float ccd_axis_limit = 1e-6f;   // Minimum axis norm for CCD
    float finite_diff_eps = 1e-3f;   // For numerical Jacobian

    int ik_type = 2;					//0 = DLS, 1 = NR
    int j_type = 1;                   //0=numerical 1=analytical
} IK_CONFIG;

float joint_limits[NUM_JOINTS][2] = {
    {-2.618, 2.168},
    {0.0, 3.14},
    {-2.967, 0},
    {-1.745, 1.745},
    {-1.22, 1.22},
    {-2.0944, 2.0944}
    /*{-M_PI, M_PI},
    {-M_PI, M_PI},
    {-M_PI, M_PI},
    {-M_PI, M_PI},
    {-M_PI, M_PI}, 
    {-M_PI, M_PI}*/
};

static const char* TAG = "InverseKinematics";

static dspm::Mat J(3, NUM_JOINTS);

static dspm::Mat T_static[NUM_JOINTS] = {dspm::Mat(4, 4), dspm::Mat(4, 4), dspm::Mat(4, 4),
                                   dspm::Mat(4, 4), dspm::Mat(4, 4), dspm::Mat(4, 4)};

static dspm::Mat Transforms[NUM_JOINTS] = {dspm::Mat(4, 4), dspm::Mat(4, 4), dspm::Mat(4, 4),
                                           dspm::Mat(4, 4), dspm::Mat(4, 4), dspm::Mat(4, 4)};

static dspm::Mat cumTransforms[NUM_JOINTS] = {dspm::Mat(4, 4), dspm::Mat(4, 4), dspm::Mat(4, 4),
                                            dspm::Mat(4, 4), dspm::Mat(4, 4), dspm::Mat(4, 4)};

static void init_T_static() {

    for (int i=0; i<NUM_JOINTS; i++) {
        float theta_x = ROT_TRANS[0][i][0];
        float theta_y = ROT_TRANS[0][i][1];
        float theta_z = ROT_TRANS[0][i][2];
        float cx = cosf(theta_x), sx = sinf(theta_x);
        float cy = cosf(theta_y), sy = sinf(theta_y);
        float cz = cosf(theta_z), sz = sinf(theta_z);
                                                   
        float Rx_data[16] = {1, 0, 0, 0,
                             0, cx, -sx, 0,
                             0, sx, cx, 0,
                             0,  0,  0,  1};
        float Ry_data[16] = {cy, 0, sy, 0,
                             0, 1,  0, 0,
                            -sy, 0, cy, 0,
                             0, 0,  0, 1};
        float Rz_data[16] = {cz, -sz, 0, 0,
                             sz, cz, 0, 0,
                             0,  0, 1, 0,
                             0,  0, 0, 1};
                                                   
        dspm::Mat Rx(Rx_data, 4, 4);
        dspm::Mat Ry(Ry_data, 4, 4);
        dspm::Mat Rz(Rz_data, 4, 4);
        T_static[i] = Rx * Ry * Rz;
        T_static[i](0, 3) = ROT_TRANS[1][i][0];
        T_static[i](1, 3) = ROT_TRANS[1][i][1];
        T_static[i](2, 3) = ROT_TRANS[1][i][2];
    }
}

static void transformationMatrix(float theta, dspm::Mat& T, int i) {
    float cz = cosf(theta), sz = sinf(theta);
    float Rz_data[16] = {cz, -sz, 0, 0,
                        sz, cz, 0,  0,
                            0,  0,  1, 0,
                            0,  0,  0, 1};
    dspm::Mat Rz(Rz_data, 4, 4);
    T = T_static[i] * Rz;
}

dspm::Mat forwardKinematics(float angles[NUM_JOINTS]) {
    dspm::Mat transform = dspm::Mat::eye(4);
    dspm::Mat T(4, 4);
    for (int i=0; i<NUM_JOINTS; i++) {
        //rotation matrix calculation
        transformationMatrix(angles[i], T, i);
        transform *= T;
        Transforms[i] = T;
    }
    return transform.getROI(0, 3, 3, 1);
}

dspm::Mat forwardKinematics_trans(int i) {
    dspm::Mat transform = dspm::Mat::eye(4);
    for (int j=0; j<i; j++) {
        transform *= Transforms[j];
    }
    return transform.getROI(0, 3, 3, 1);
}

static void init_cumTransforms() {
    dspm::Mat transform = dspm::Mat::eye(4);
    for (int i=0; i<NUM_JOINTS; i++) {
        transform *= Transforms[i];
        cumTransforms[i] = transform;
    }
}

void numericalJacobian(const float angles[NUM_JOINTS], const dspm::Mat& pos) {

    const float eps = IK_CONFIG.finite_diff_eps;
    dspm::Mat perturbed_pos(3, 1);
    dspm::Mat j_row(3, 1);
    dspm::Mat T_original(4, 4);

    for (int i = 0; i < NUM_JOINTS; i++) {
        float theta = angles[i];
        T_original = Transforms[i];

        theta += eps;
        transformationMatrix(theta, Transforms[i], i);
        perturbed_pos = forwardKinematics_trans(NUM_JOINTS);
        j_row = (perturbed_pos - pos) / eps;
        J(0, i) = j_row(0, 0);
        J(1, i) = j_row(1, 0);
        J(2, i) = j_row(2, 0);

        Transforms[i] = T_original;
    }
}

void analyticalJacobian(const dspm::Mat& pos) {

    init_cumTransforms();
    
    for (int i = 0; i < NUM_JOINTS; i++) {

        dspm::Mat curr_t = cumTransforms[i];

        // Get joint position (translation part of transform)
        dspm::Mat joint_pos(3, 1);
        joint_pos(0, 0) = curr_t(0, 3);
        joint_pos(1, 0) = curr_t(1, 3);
        joint_pos(2, 0) = curr_t(2, 3);
        
        // Compute vector from joint to end-effector
        dspm::Mat ee_to_joint = pos - joint_pos;
        
        // Compute Jacobian columns for each rotation axis
        
        // Z rotation
        J(0, i) = curr_t(1, 2) * ee_to_joint(2, 0) - curr_t(2, 2) * ee_to_joint(1, 0);
        J(1, i) = curr_t(2, 2) * ee_to_joint(0, 0) - curr_t(0, 2) * ee_to_joint(2, 0);
        J(2, i) = curr_t(0, 2) * ee_to_joint(1, 0) - curr_t(1, 2) * ee_to_joint(0, 0);
    }
}

bool inverse_kinematics_NR(const dspm::Mat& target_pos, 
    const float current_angles[NUM_JOINTS], 
    float result[NUM_JOINTS]) {

    float angles[NUM_JOINTS];
    memcpy(angles, current_angles, sizeof(float) * NUM_JOINTS);

    dspm::Mat J_T(NUM_JOINTS, 3);
    dspm::Mat J_JT(3, 3);
    dspm::Mat damping(3, 3);
    dspm::Mat inv_J_JT(3, 3);
    dspm::Mat delta_angles(NUM_JOINTS, 1);
    dspm::Mat position(3, 1);
    dspm::Mat error(3, 1);
    dspm::Mat new_angles(1, NUM_JOINTS);

    for (int iter = 0; iter < IK_CONFIG.max_iterations; iter++) {
        
        position = forwardKinematics(angles);
        error = target_pos - position;
        float error_norm = error.norm();

        // Check convergence
        if (error_norm < IK_CONFIG.tolerance) {
            memcpy(result, angles, sizeof(float) * NUM_JOINTS);
            ESP_LOGI(TAG, "Converged after %d iterations (error=%.6f)", iter, error_norm);
            return true;
        }

        // Compute Jacobian
        if (!IK_CONFIG.j_type) {
            numericalJacobian(angles, position);
        } else {
            analyticalJacobian(position);
        }

        J_T = J.t();
        J_JT = J * J_T;

        if (IK_CONFIG.ik_type == 0) {
            // Damped Least Squares solution
            damping = dspm::Mat::eye(3) * IK_CONFIG.dls_damping * IK_CONFIG.dls_damping;
            inv_J_JT = (J_JT + damping).pinv();
            delta_angles = J_T * inv_J_JT * error;
        } else {
            // Standard Newton-Raphson solution
            inv_J_JT = J_JT.pinv();
            delta_angles = J_T * inv_J_JT * error;
        }

        // Update angles with appropriate step size
        float step_size = (IK_CONFIG.ik_type == 0) ? IK_CONFIG.dls_step_size : IK_CONFIG.nr_step_size;
        new_angles = step_size * delta_angles.t();

        // Apply joint limits
        for (int i = 0; i < NUM_JOINTS; i++) {
            float new_val = angles[i] + new_angles(0, i);
            angles[i] = fmaxf(joint_limits[i][0], new_val);
            angles[i] = fminf(joint_limits[i][1], new_val);
        }
    }
    position = forwardKinematics(angles);
    float final_error = (target_pos - position).norm();
    memcpy(result, angles, sizeof(float) * NUM_JOINTS);
    return final_error < IK_CONFIG.tolerance;
}

bool inverse_kinematics_CCD(const dspm::Mat& target_pos,
                          const float current_angles[NUM_JOINTS],
                          float result[NUM_JOINTS]) {

    float angles[NUM_JOINTS];
    memcpy(angles, current_angles, sizeof(float) * NUM_JOINTS);
    dspm::Mat ee_pos(3, 1);
    dspm::Mat error(3, 1);

    forwardKinematics(angles); //set transforms

    for (int iter = 0; iter < IK_CONFIG.max_iterations; iter++) {
        init_cumTransforms();
        ee_pos = cumTransforms[NUM_JOINTS-1].getROI(0, 3, 3, 1);
        error = target_pos - ee_pos;
        float current_error = error.norm();

        // Check convergence
        if (current_error < IK_CONFIG.tolerance) {
            memcpy(result, angles, sizeof(float) * NUM_JOINTS);
            ESP_LOGI(TAG, "Converged after %d iterations (error=%.6f)", iter, current_error);
            return true;
        }

        // Process joints from end-effector to base
        for (int i = NUM_JOINTS - 1; i >= 0; i--) {
            // Compute joint position
            dspm::Mat joint_pos = cumTransforms[i].getROI(0, 3, 3, 1);

            // Calculate vectors
            dspm::Mat to_ee = ee_pos - joint_pos;
            dspm::Mat to_target = target_pos - joint_pos;

            // Normalize vectors
            float to_ee_norm = to_ee.norm();
            float to_target_norm = to_target.norm();

            if (to_ee_norm < IK_CONFIG.ccd_epsilon || to_target_norm < IK_CONFIG.ccd_epsilon) {
                continue;
            }

            to_ee = to_ee / to_ee_norm;
            to_target = to_target / to_target_norm;

            // Calculate rotation axis and angle
            dspm::Mat rotation_axis(3, 1);
            rotation_axis(0,0) = to_ee(1,0)*to_target(2,0) - to_ee(2,0)*to_target(1,0);
            rotation_axis(1,0) = to_ee(2,0)*to_target(0,0) - to_ee(0,0)*to_target(2,0);
            rotation_axis(2,0) = to_ee(0,0)*to_target(1,0) - to_ee(1,0)*to_target(0,0);
            
            float axis_norm = rotation_axis.norm();

            if (axis_norm < IK_CONFIG.ccd_axis_limit) {
                continue;
            }

            rotation_axis = rotation_axis / axis_norm;

            // Calculate rotation angle
            float dot_product = dspm::Mat::dotProduct(to_ee, to_target); 
            float rotation_angle = atan2f(axis_norm, dot_product);

            dspm::Mat joint_rot = cumTransforms[i].getROI(0, 0, 3, 3);
            
            dspm::Mat local_axis = joint_rot.t() * rotation_axis;
            
            float angle_update = rotation_angle * local_axis(2, 0);
            float new_angle = angles[i] + angle_update;
            
            // Apply joint limits
            new_angle = fmaxf(joint_limits[i][0], new_angle);
            new_angle = fminf(joint_limits[i][1], new_angle);
            
            angles[i] = new_angle;
            
            transformationMatrix(angles[i], Transforms[i], i);
        }
    }
    // Return final solution
    ee_pos = forwardKinematics_trans(NUM_JOINTS);

    float final_error = (target_pos - ee_pos).norm();

    memcpy(result, angles, sizeof(float) * NUM_JOINTS);

    return final_error < IK_CONFIG.tolerance;
}

bool calculate_ik(const dspm::Mat& target_pos, 
const float current_angles[NUM_JOINTS], 
float result[NUM_JOINTS]) {
    init_T_static();
    if (IK_CONFIG.ik_type == 0 || IK_CONFIG.ik_type == 1) {
        return inverse_kinematics_NR(target_pos, current_angles, result);
    } else {
        return inverse_kinematics_CCD(target_pos, current_angles, result);
    }
}
volatile bool stop_cpu_monitor = false;


void calculate_cpu_usage() {
    static uint64_t prev_time = 0;
    static uint32_t prev_idle_time0 = 0;
    static uint32_t prev_idle_time1 = 0;

    UBaseType_t num_tasks = uxTaskGetNumberOfTasks();
    TaskStatus_t *task_status = (TaskStatus_t *)malloc(num_tasks * sizeof(TaskStatus_t));
    if (task_status == nullptr) return;

    num_tasks = uxTaskGetSystemState(task_status, num_tasks, nullptr);

    uint32_t idle_time0 = 0, idle_time1 = 0;
    for (UBaseType_t i = 0; i < num_tasks; i++) {
        if (strcmp(task_status[i].pcTaskName, "IDLE0") == 0) {
            idle_time0 = task_status[i].ulRunTimeCounter;
        } else if (strcmp(task_status[i].pcTaskName, "IDLE1") == 0) {
            idle_time1 = task_status[i].ulRunTimeCounter;
        }
    }

    uint64_t current_time = esp_timer_get_time();
    if (prev_time != 0 && current_time > prev_time) {
        uint64_t elapsed_time = current_time - prev_time;

        uint32_t delta_idle0 = idle_time0 - prev_idle_time0;
        uint32_t delta_idle1 = idle_time1 - prev_idle_time1;

        float cpu0 = 100.0f - (delta_idle0 * 100.0f / elapsed_time);
        float cpu1 = 100.0f - (delta_idle1 * 100.0f / elapsed_time);

        printf("CPU0 Usage: %.2f%%, CPU1 Usage: %.2f%%\n", cpu0, cpu1);
    }

    prev_time = current_time;
    prev_idle_time0 = idle_time0;
    prev_idle_time1 = idle_time1;
    free(task_status);
}
void cpu_monitor_task(void *pvParameters) {
    while (!stop_cpu_monitor) {  // Exit when flag is set
        calculate_cpu_usage();
        vTaskDelay(pdMS_TO_TICKS(600));
    }
    vTaskDelete(NULL);  // Delete this task
}
void print_heap_stats() {
    multi_heap_info_t info;
    heap_caps_get_info(&info, MALLOC_CAP_DEFAULT);
    ESP_LOGI("HEAP", "Total free heap: %d bytes", info.total_free_bytes);
    ESP_LOGI("HEAP", "Largest free block: %d bytes", info.largest_free_block);
    ESP_LOGI("HEAP", "Minimum free heap ever: %d bytes", info.minimum_free_bytes);
}
void print_stack_usage() {
    TaskHandle_t current_task = xTaskGetCurrentTaskHandle();
    const char* task_name = pcTaskGetName(current_task);
    UBaseType_t high_water_mark = uxTaskGetStackHighWaterMark(current_task);

    ESP_LOGI("STACK", "Task \"%s\" stack high watermark: %u bytes free", task_name, high_water_mark * sizeof(StackType_t));
}
void print_cpu_usage() {
    char buffer[1024]; // Large buffer to hold stats
    vTaskGetRunTimeStats(buffer);

    ESP_LOGI("CPU", "Task CPU usage:\n%s", buffer);
}

char runtime_stats_before[1024];
char runtime_stats_after[1024];
struct TaskRunTime {
    uint32_t run_time;
    float percentage;
};

std::map<std::string, TaskRunTime> parse_runtime_stats(const char *stats) {
    std::map<std::string, TaskRunTime> task_map;
    std::istringstream ss(stats);
    std::string line;
    while (std::getline(ss, line)) {
        std::istringstream line_ss(line);
        std::string task_name;
        uint32_t run_time = 0;
        float percentage = 0;

        line_ss >> task_name >> run_time >> percentage;
        if (!task_name.empty()) {
            task_map[task_name] = {run_time, percentage};
        }
    }
    return task_map;
}

void print_cpu_usage_delta(const char *before, const char *after) {
    auto map_before = parse_runtime_stats(before);
    auto map_after = parse_runtime_stats(after);

    uint32_t total_before = 0, total_after = 0;
    for (auto &p : map_before) total_before += p.second.run_time;
    for (auto &p : map_after) total_after += p.second.run_time;

    uint32_t total_delta = total_after - total_before;

    printf("Task CPU usage during test:\n");
    for (auto &p : map_after) {
        const std::string &task = p.first;
        uint32_t run_after = p.second.run_time;
        uint32_t run_before = map_before.count(task) ? map_before[task].run_time : 0;
        uint32_t delta = run_after - run_before;

        float cpu_percent = total_delta > 0 ? (100.0f * delta / total_delta) : 0.0f;
        printf("%-15s: %10lu (%.2f%%)\n", task.c_str(), (unsigned long)delta, cpu_percent);

    }

}
struct HeapSnapshot {
    size_t total_free;
    size_t min_free;
    size_t largest_block;
};
HeapSnapshot get_heap_snapshot() {
    multi_heap_info_t info;
    heap_caps_get_info(&info, MALLOC_CAP_DEFAULT);
    
    HeapSnapshot snapshot;
    snapshot.total_free = info.total_free_bytes;
    snapshot.min_free = info.minimum_free_bytes;
    snapshot.largest_block = info.largest_free_block;
    return snapshot;
}


void print_heap_delta(const HeapSnapshot& before, const HeapSnapshot& after) {
    ESP_LOGI("HEAP_DELTA", "Heap Usage Delta:");
    ESP_LOGI("HEAP_DELTA", "  Total Free Heap:    %d -> %d (%d bytes used)", 
             (int)before.total_free, (int)after.total_free,
             (int)(before.total_free - after.total_free));
    ESP_LOGI("HEAP_DELTA", "  Minimum Ever Free:  %d -> %d (%d bytes drop)",
             (int)before.min_free, (int)after.min_free,
             (int)(before.min_free - after.min_free));
    ESP_LOGI("HEAP_DELTA", "  Largest Free Block: %d -> %d (%d bytes drop)",
             (int)before.largest_block, (int)after.largest_block,
             (int)(before.largest_block - after.largest_block));
}



void test_ik_with_multiple_targets() {
    vTaskGetRunTimeStats(runtime_stats_before);
    ESP_LOGI(TAG, "Starting IK test with multiple target positions");
    
    // Initial joint angles (home position)
    float initial_angles[NUM_JOINTS] = {
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f
    };

	print_heap_stats();
	print_stack_usage();
	float test_positions[7][3] = {
   	 	{ 0.3f,  0.0f,  0.3f},   // Right front
    	{-0.3f,  0.0f,  0.3f},   // Left front
    	{ 0.0f,  0.3f,  0.3f},   // Forward
    	{ 0.0f, -0.3f,  0.3f},   // Backward
    	{ 0.0f,  0.0f,  0.5f},   // Upper center (higher than before)
    	{ 0.3f,  0.3f,  0.3f},   // Upper right front corner
   		 {-0.3f, -0.3f,  0.3f}    // Lower left back corner
	};
    float result_angles[NUM_JOINTS];
    dspm::Mat final_pos(3, 1);

    for (int i = 0; i < 7; i++) {
   
        ESP_LOGI(TAG, "\n===== TEST %d ======", i+1);
        
        // Create target position matrix
        dspm::Mat target_pos(test_positions[i], 3, 1);
        ESP_LOGI(TAG, "Target Position: [%.3f, %.3f, %.3f]", 
                 target_pos(0,0), target_pos(1,0), target_pos(2,0));

        // Reset to initial angles for each test
        memcpy(result_angles, initial_angles, sizeof(float) * NUM_JOINTS);
		
		int64_t start_time = esp_timer_get_time();
		HeapSnapshot heap_before = get_heap_snapshot();
		vTaskGetRunTimeStats(runtime_stats_before);
	
        // Calculate IK
        bool success = calculate_ik(target_pos, initial_angles, result_angles);
        vTaskDelay(pdMS_TO_TICKS(100));  // Let tasks settle
		vTaskGetRunTimeStats(runtime_stats_after);
        int64_t end_time = esp_timer_get_time();
        int64_t elapsed_us = end_time - start_time;
       	HeapSnapshot heap_after = get_heap_snapshot();

		print_heap_delta(heap_before, heap_after);

		print_cpu_usage_delta(runtime_stats_before, runtime_stats_after);
        ESP_LOGI(TAG, "IK calculation took %lld microseconds", elapsed_us);
        // Print results
        if (success) {
            ESP_LOGI(TAG, "IK converged successfully!");
        } else {
            ESP_LOGW(TAG, "IK did not fully converge");
        }
		
        // Print final joint angles
        ESP_LOGI(TAG, "Final Joint Angles:");
        for (int j = 0; j < NUM_JOINTS; j++) {
            ESP_LOGI(TAG, "Joint %d: Z=%.6f", j+1,
                    result_angles[j]);
        }
		print_heap_stats();
		print_stack_usage();
		
		vTaskGetRunTimeStats(runtime_stats_after);

		print_heap_delta(heap_before, heap_after);
		print_cpu_usage_delta(runtime_stats_before, runtime_stats_after);
	
        int iterations = 0;



        if (success) {
            ESP_LOGI(TAG, "IK converged in %d iterations.", iterations);
        } else {
            ESP_LOGI(TAG, "IK did not converge after max iterations (%d).", iterations);
        }
        // Calculate achieved position
        final_pos = forwardKinematics(result_angles);
        
        // Print achieved vs target
        ESP_LOGI(TAG, "Achieved Position: [%.6f, %.6f, %.6f]",
                 final_pos(0,0), final_pos(1,0), final_pos(2,0));
        ESP_LOGI(TAG, "Target Position:   [%.6f, %.6f, %.6f]",
                 target_pos(0,0), target_pos(1,0), target_pos(2,0));

        // Calculate position error
        dspm::Mat error = target_pos - final_pos;
        float position_error = error.norm();
        ESP_LOGI(TAG, "Position Error: %.6f meters", position_error);
    }

    ESP_LOGI(TAG, "IK testing completed");
 
}


	
extern "C" void app_main() {
     ESP_LOGI(TAG, "Starting Inverse Kinematics Test Suite");
    ESP_LOGI(TAG, "System Configuration:");
    ESP_LOGI(TAG, "- Number of joints: %d", NUM_JOINTS);
    ESP_LOGI(TAG, "- IK Method: %s", 
            IK_CONFIG.ik_type == 0 ? "DLS" : 
            IK_CONFIG.ik_type == 1 ? "NR" : "CCD");
    ESP_LOGI(TAG, "- Jacobian: %s", 
            IK_CONFIG.j_type == 0 ? "Numerical" : "Analytical");

    // Initialize static transforms (must be done once before IK calculations)
    init_T_static();
	
    // Start CPU monitor task
    stop_cpu_monitor = false;
    xTaskCreate(cpu_monitor_task, "CPU_Monitor_Task", 2048, NULL, 5, NULL);
	
    // Run IK test
    test_ik_with_multiple_targets();
	vTaskDelay(pdMS_TO_TICKS(1000));


    // Signal the CPU monitor to stop and wait a bit to clean up
    stop_cpu_monitor = true;
    vTaskDelay(pdMS_TO_TICKS(1000));

    ESP_LOGI(TAG, "Application finished.");
}
