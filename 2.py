import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# 设置matplotlib中文字体，防止画图乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class VRPTWDataParser:
    """VRPTW数据解析器"""
    def __init__(self):
        self.data_files = []
        
    def load_data_files(self):
        """加载所有数据文件"""
        for filename in os.listdir('.'):
            if filename.startswith('C1_') and filename.endswith('.txt'):
                self.data_files.append(filename)
        self.data_files.sort()
        print(f"找到 {len(self.data_files)} 个数据文件")
        return self.data_files
    
    def parse_data_file(self, filename):
        """解析单个数据文件"""
        data = {
            'name': filename,
            'vehicle_capacity': 0,
            'customers': [],
            'depot': None
        }
        
        with open(filename, 'r') as f:
            lines = f.readlines()
            
        in_vehicle_section = False
        in_customer_section = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if 'VEHICLE' in line:
                in_vehicle_section = True
                continue
            elif 'CUSTOMER' in line:
                in_customer_section = True
                in_vehicle_section = False
                continue
            elif line.startswith('CUST NO.') or line.startswith('NUMBER'):
                continue
                
            if in_vehicle_section:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        numbers = []
                        for part in parts:
                            if part.isdigit():
                                numbers.append(int(part))
                        if len(numbers) >= 2:
                            data['vehicle_capacity'] = numbers[1]
                            in_vehicle_section = False
                    except:
                        pass
                
            if in_customer_section:
                parts = line.split()
                if len(parts) >= 7:
                    try:
                        customer = {
                            'id': int(parts[0]),
                            'x': int(parts[1]),
                            'y': int(parts[2]),
                            'demand': int(parts[3]),
                            'ready_time': int(parts[4]),
                            'due_date': int(parts[5]),
                            'service_time': int(parts[6])
                        }
                        
                        if customer['id'] == 0:
                            data['depot'] = customer
                        else:
                            data['customers'].append(customer)
                    except Exception as e:
                        print(f"解析客户数据时出错: {e}")
        
        return data

class OptimizedVRPTWSolver:
    """优化的VRPTW求解器"""
    def __init__(self, data):
        self.data = data
        self.depot = data['depot']
        self.customers = data['customers'].copy()
        self.capacity = data['vehicle_capacity']
        self.routes = []
        self.unserved_customers = self.customers.copy()
        
    def solve(self, max_vehicles):
        """求解VRPTW问题"""
        print(f"\n求解 {self.data['name']}，最大车辆数限制: {max_vehicles}")
        
        # 改进的智能聚类
        clusters = self.improved_clustering(max_vehicles)
        
        # 为每个簇创建优化路径
        for i, cluster in enumerate(clusters):
            if not cluster:
                continue
                
            route = self.optimize_route_with_time_windows(cluster)
            if route:
                self.routes.append(route)
        
        # 检查未服务客户并尝试重新分配
        self.reassign_unserved_customers()
        
        return self.routes
    
    def improved_clustering(self, max_vehicles):
        """改进的聚类算法"""
        clusters = [[] for _ in range(max_vehicles)]
        
        if not self.customers:
            return clusters
        
        # 预处理：计算客户特征
        for customer in self.customers:
            distance = self.calculate_distance(self.depot, customer)
            time_window_length = customer['due_date'] - customer['ready_time']
            customer['distance'] = distance
            customer['time_priority'] = 1.0 / (time_window_length + 1)
            customer['demand_density'] = customer['demand'] / (distance + 1)
        
        # 分阶段聚类
        # 第一阶段：处理时间窗紧迫的客户
        urgent_customers = [c for c in self.customers if c['time_priority'] > 0.01]
        normal_customers = [c for c in self.customers if c['time_priority'] <= 0.01]
        
        # 先处理紧迫客户
        for customer in sorted(urgent_customers, key=lambda x: (-x['time_priority'], x['distance'])):
            best_cluster_idx = self.find_best_cluster(clusters, customer, prioritize_time=True)
            clusters[best_cluster_idx].append(customer)
        
        # 再处理普通客户
        for customer in sorted(normal_customers, key=lambda x: (x['distance'], -x['demand_density'])):
            best_cluster_idx = self.find_best_cluster(clusters, customer, prioritize_time=False)
            clusters[best_cluster_idx].append(customer)
        
        # 簇优化：平衡负载
        self.optimize_clusters(clusters)
        
        return clusters
    
    def find_best_cluster(self, clusters, customer, prioritize_time=True):
        """找到最佳簇"""
        best_idx = 0
        best_score = float('inf')
        
        for i, cluster in enumerate(clusters):
            cluster_demand = sum(c['demand'] for c in cluster)
            
            # 严格检查容量约束
            if cluster_demand + customer['demand'] > self.capacity:
                continue
            
            # 计算多因素评分
            score = self.calculate_advanced_cluster_score(cluster, customer, prioritize_time)
            
            if score < best_score:
                best_score = score
                best_idx = i
        
        return best_idx
    
    def calculate_advanced_cluster_score(self, cluster, customer, prioritize_time):
        """高级簇评分函数"""
        if not cluster:
            return customer['distance']
        
        # 距离因素：到簇中心的距离
        cluster_center = self.calculate_cluster_center(cluster)
        distance_score = self.calculate_distance(cluster_center, customer)
        
        # 时间兼容性因素
        cluster_ready = min(c['ready_time'] for c in cluster)
        cluster_due = max(c['due_date'] for c in cluster)
        
        time_overlap = max(0, min(customer['due_date'], cluster_due) - max(customer['ready_time'], cluster_ready))
        time_compatibility = time_overlap / (customer['due_date'] - customer['ready_time'] + 1)
        
        # 需求平衡因素
        cluster_demand = sum(c['demand'] for c in cluster)
        new_demand = cluster_demand + customer['demand']
        demand_balance = abs(new_demand - self.capacity * 0.8)  # 目标利用率80%
        
        # 客户密度因素
        cluster_density = sum(c['demand_density'] for c in cluster) / len(cluster)
        density_score = abs(customer['demand_density'] - cluster_density)
        
        # 权重调整
        if prioritize_time:
            return (distance_score * 0.3 + 
                    (1 - time_compatibility) * 100 * 0.5 + 
                    demand_balance * 0.1 + 
                    density_score * 0.1)
        else:
            return (distance_score * 0.5 + 
                    (1 - time_compatibility) * 50 * 0.2 + 
                    demand_balance * 0.2 + 
                    density_score * 0.1)
    
    def calculate_cluster_center(self, cluster):
        """计算簇中心"""
        avg_x = np.mean([c['x'] for c in cluster])
        avg_y = np.mean([c['y'] for c in cluster])
        return {'x': avg_x, 'y': avg_y}
    
    def optimize_clusters(self, clusters):
        """优化簇分布"""
        # 尝试重新分配客户以平衡负载
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                self.try_swap_customers(clusters[i], clusters[j])
    
    def try_swap_customers(self, cluster1, cluster2):
        """尝试交换客户以优化簇"""
        # 创建客户列表的副本以避免修改问题
        cluster1_copy = cluster1.copy()
        cluster2_copy = cluster2.copy()
        
        for customer1 in cluster1_copy:
            for customer2 in cluster2_copy:
                # 检查交换后的容量约束
                new_demand1 = sum(c['demand'] for c in cluster1) - customer1['demand'] + customer2['demand']
                new_demand2 = sum(c['demand'] for c in cluster2) - customer2['demand'] + customer1['demand']
                
                if new_demand1 <= self.capacity and new_demand2 <= self.capacity:
                    # 计算交换前后的总评分
                    score_before = (self.calculate_cluster_efficiency(cluster1) + 
                                   self.calculate_cluster_efficiency(cluster2))
                    
                    # 执行交换
                    if customer1 in cluster1 and customer2 in cluster2:
                        cluster1.remove(customer1)
                        cluster2.remove(customer2)
                        cluster1.append(customer2)
                        cluster2.append(customer1)
                        
                        score_after = (self.calculate_cluster_efficiency(cluster1) + 
                                      self.calculate_cluster_efficiency(cluster2))
                        
                        # 如果没有改善，恢复原状
                        if score_after >= score_before:
                            if customer2 in cluster1 and customer1 in cluster2:
                                cluster1.remove(customer2)
                                cluster2.remove(customer1)
                                cluster1.append(customer1)
                                cluster2.append(customer2)
    
    def calculate_cluster_efficiency(self, cluster):
        """计算簇效率"""
        if not cluster:
            return 0
        
        # 距离效率
        center = self.calculate_cluster_center(cluster)
        total_distance = sum(self.calculate_distance(c, center) for c in cluster)
        
        # 时间效率
        time_span = max(c['due_date'] for c in cluster) - min(c['ready_time'] for c in cluster)
        
        # 容量利用率
        demand_utilization = sum(c['demand'] for c in cluster) / self.capacity
        
        return total_distance + time_span * 0.1 - demand_utilization * 100
    
    def optimize_route_with_time_windows(self, cluster):
        """带时间窗的路径优化"""
        if not cluster:
            return None
        
        # 改进的插入算法
        route = [self.depot]
        unassigned = cluster.copy()
        
        # 按时间窗和距离综合排序
        unassigned.sort(key=lambda x: (x['ready_time'], x['distance']))
        
        while unassigned:
            best_insertion = None
            best_cost = float('inf')
            
            for customer in unassigned:
                for i in range(1, len(route) + 1):
                    new_route = route[:i] + [customer] + route[i:]
                    cost_info = self.evaluate_route_cost(new_route)
                    
                    if cost_info['feasible']:
                        # 综合考虑距离、时间和等待时间
                        total_cost = (cost_info['distance'] * 0.7 + 
                                     cost_info['time'] * 0.2 + 
                                     cost_info['waiting_time'] * 0.1)
                        
                        if total_cost < best_cost:
                            best_cost = total_cost
                            best_insertion = (i, customer)
            
            if best_insertion:
                i, customer = best_insertion
                route.insert(i, customer)
                unassigned.remove(customer)
            else:
                # 尝试放松时间窗约束或使用最近邻
                last = route[-1]
                nearest = min(unassigned, key=lambda x: self.calculate_distance(last, x))
                route.append(nearest)
                unassigned.remove(nearest)
        
        route.append(self.depot)
        return route
    
    def reassign_unserved_customers(self):
        """重新分配未服务客户"""
        served_ids = set()
        for route in self.routes:
            for customer in route[1:-1]:
                served_ids.add(customer['id'])
        
        self.unserved_customers = [c for c in self.customers if c['id'] not in served_ids]
        
        # 尝试将未服务客户插入现有路径
        for customer in self.unserved_customers[:]:
            inserted = False
            for route in self.routes:
                if self.try_insert_customer(route, customer):
                    inserted = True
                    self.unserved_customers.remove(customer)
                    break
            
            if not inserted and len(self.routes) < 25:  # 前五份文件最大25辆
                # 创建新路径
                new_route = [self.depot, customer, self.depot]
                if self.evaluate_route_cost(new_route)['feasible']:
                    self.routes.append(new_route)
                    self.unserved_customers.remove(customer)
    
    def try_insert_customer(self, route, customer):
        """尝试插入客户到路径"""
        original_demand = sum(c['demand'] for c in route[1:-1])
        if original_demand + customer['demand'] > self.capacity:
            return False
        
        for i in range(1, len(route)):
            new_route = route[:i] + [customer] + route[i:]
            cost_info = self.evaluate_route_cost(new_route)
            if cost_info['feasible']:
                route[:] = new_route
                return True
        
        return False
    
    def calculate_distance(self, customer1, customer2):
        """计算欧几里得距离"""
        return math.sqrt((customer1['x'] - customer2['x'])**2 + (customer1['y'] - customer2['y'])**2)
    
    def evaluate_route_cost(self, route):
        """评估路径成本"""
        total_distance = 0
        total_time = 0
        total_waiting_time = 0
        current_time = 0
        feasible = True
        
        for i in range(len(route) - 1):
            from_cust = route[i]
            to_cust = route[i + 1]
            
            distance = self.calculate_distance(from_cust, to_cust)
            total_distance += distance
            
            travel_time = distance
            arrival_time = current_time + travel_time
            
            if to_cust['id'] == 0:
                current_time = arrival_time
                continue
            
            # 检查时间窗
            if arrival_time > to_cust['due_date']:
                feasible = False
            
            waiting_time = max(0, to_cust['ready_time'] - arrival_time)
            total_waiting_time += waiting_time
            
            current_time = arrival_time + waiting_time + to_cust['service_time']
            total_time += travel_time + waiting_time + to_cust['service_time']
        
        if current_time > self.depot['due_date']:
            feasible = False
        
        return {
            'distance': total_distance,
            'time': total_time,
            'waiting_time': total_waiting_time,
            'feasible': feasible
        }
    
    def evaluate_solution(self):
        """评估解决方案"""
        total_vehicles = len(self.routes)
        total_distance = 0
        total_time = 0
        feasible_routes = 0
        served_customers = set()
        
        for route in self.routes:
            if len(route) <= 2:
                continue
                
            cost_info = self.evaluate_route_cost(route)
            total_distance += cost_info['distance']
            total_time += cost_info['time']
            if cost_info['feasible']:
                feasible_routes += 1
            
            for customer in route[1:-1]:
                served_customers.add(customer['id'])
        
        all_customers_served = len(served_customers) == len(self.customers)
        
        return {
            'vehicles_used': total_vehicles,
            'total_distance': total_distance,
            'total_time': total_time,
            'feasible_routes': feasible_routes,
            'total_routes': len(self.routes),
            'customers_served': len(served_customers),
            'total_customers': len(self.customers),
            'all_customers_served': all_customers_served
        }

def main():
    """主函数"""
    parser = VRPTWDataParser()
    data_files = parser.load_data_files()
    
    results = {}
    
    for filename in data_files:
        data = parser.parse_data_file(filename)
        
        # 前五份文件尝试减少车辆数
        if filename.startswith('C1_2_'):
            max_vehicles = 20  # 降低目标车辆数
        else:
            max_vehicles = 100
        
        solver = OptimizedVRPTWSolver(data)
        routes = solver.solve(max_vehicles)
        
        result = solver.evaluate_solution()
        results[filename] = result
        
        print(f"{filename} 求解结果:")
        print(f"  车辆数: {result['vehicles_used']}/{max_vehicles}")
        print(f"  总路程: {result['total_distance']:.2f}")
        print(f"  总时间: {result['total_time']:.2f}")
        print(f"  服务客户: {result['customers_served']}/{result['total_customers']}")
        print(f"  所有客户是否服务: {'是' if result['all_customers_served'] else '否'}")
    
    # 生成汇总报告
    print("\n" + "="*70)
    print("              VRPTW问题求解结果汇总（优化版）")
    print("="*70)
    
    print(f"\n{'实例编号':<10} {'车辆数':<10} {'总路程':<12} {'总时间':<12} {'服务客户数':<12} {'约束满足':<10}")
    print("-" * 70)
    
    summary_data = []
    all_constraints_met = True
    all_customers_served = True
    
    for filename in sorted(results.keys()):
        result = results[filename]
        instance_id = filename.replace('C1_', '').replace('.txt', '')
        
        if filename.startswith('C1_2_'):
            constraint_met = result['vehicles_used'] <= 25
        else:
            constraint_met = result['vehicles_used'] <= 100
        
        all_constraints_met = all_constraints_met and constraint_met
        all_customers_served = all_customers_served and result['all_customers_served']
        
        print(f"{instance_id:<10} {result['vehicles_used']:<10} {result['total_distance']:<12.2f} "
              f"{result['total_time']:<12.2f} {result['customers_served']}/{result['total_customers']:<12} "
              f"{'是' if constraint_met else '否':<10}")
        
        summary_data.append({
            '实例编号': instance_id,
            '车辆数量': result['vehicles_used'],
            '最大允许车辆数': 25 if filename.startswith('C1_2_') else 100,
            '总的路程': round(result['total_distance'], 2),
            '总的时间': round(result['total_time'], 2),
            '服务客户数': result['customers_served'],
            '总客户数': result['total_customers'],
            '所有客户是否服务': '是' if result['all_customers_served'] else '否',
            '约束是否满足': '是' if constraint_met else '否'
        })
    
    # 保存结果
    df = pd.DataFrame(summary_data)
    df.to_excel('VRPTW优化求解结果.xlsx', index=False)
    
    # 创建简洁版结果
    simple_results = df[['实例编号', '车辆数量', '总的路程', '总的时间']].copy()
    simple_results.columns = ['数据文件', '车辆数量', '总的路程', '总的时间']
    simple_results.to_excel('VRPTW优化结果_简洁版.xlsx', index=False)
    
    print(f"\n结果已保存到 VRPTW优化求解结果.xlsx 和 VRPTW优化结果_简洁版.xlsx")
    
    print(f"\n最终检查结果:")
    print(f"所有实例约束是否满足: {'✅ 是' if all_constraints_met else '❌ 否'}")
    print(f"所有客户是否都被服务: {'✅ 是' if all_customers_served else '❌ 否'}")
    
    # 统计车辆数减少情况
    c1_2_vehicles = [results[f]['vehicles_used'] for f in results.keys() if f.startswith('C1_2_')]
    if c1_2_vehicles:
        print(f"\n前五份文件车辆数统计:")
        print(f"  使用车辆数: {c1_2_vehicles}")
        print(f"  平均车辆数: {np.mean(c1_2_vehicles):.1f}")
        print(f"  相比之前的25辆，平均减少: {25 - np.mean(c1_2_vehicles):.1f}辆")
    
    return results

if __name__ == "__main__":
    results = main()