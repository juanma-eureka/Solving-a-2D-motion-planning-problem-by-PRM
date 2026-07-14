import numpy as np
import pylab as pl
import sys
import math

sys.path.append('osr_examples/scripts/')
import environment_2d

np.random.seed(4)
env = environment_2d.Environment(10, 6, 5)

def is_segment_collision_free(env, x1, y1, x2, y2, num_samples=30):
    for i in range(num_samples + 1):
        t = i / float(num_samples)
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        if env.check_collision(x, y): return False
    return True

def solve_prm(env, q, num_samples=400, radius=2.5):
    x_start, y_start, x_goal, y_goal = q
    vertices = [(x_start, y_start), (x_goal, y_goal)]
    while len(vertices) < num_samples:
        x, y = np.random.uniform(0, 10), np.random.uniform(0, 6)
        if not env.check_collision(x, y): vertices.append((x, y))
    adj = {i: [] for i in range(len(vertices))}
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            dist = math.hypot(vertices[j][0] - vertices[i][0], vertices[j][1] - vertices[i][1])
            if dist <= radius and is_segment_collision_free(env, vertices[i][0], vertices[i][1], vertices[j][0], vertices[j][1]):
                adj[i].append(j); adj[j].append(i)
    queue, visited, found = [0], {0: None}, False
    while queue:
        curr = queue.pop(0)
        if curr == 1: found = True; break
        for n in adj[curr]:
            if n not in visited: visited[n] = curr; queue.append(n)
    if found:
        path, curr = [], 1
        while curr is not None: path.append(vertices[curr]); curr = visited[curr]
        return path[::-1]
    return None

def solve_rrt(env, q, max_iter=3000, step_size=0.4, goal_bias=0.1):
    x_start, y_start, x_goal, y_goal = q
    nodes, parents = [(x_start, y_start)], {0: None}
    for _ in range(max_iter):
        q_rand = (x_goal, y_goal) if np.random.rand() < goal_bias else (np.random.uniform(0, 10), np.random.uniform(0, 6))
        dists = [math.hypot(n[0] - q_rand[0], n[1] - q_rand[1]) for n in nodes]
        nearest_idx = np.argmin(dists)
        q_near = nodes[nearest_idx]
        theta = math.atan2(q_rand[1] - q_near[1], q_rand[0] - q_near[0])
        q_new = (q_near[0] + min(step_size, dists[nearest_idx]) * math.cos(theta), q_near[1] + min(step_size, dists[nearest_idx]) * math.sin(theta))
        if is_segment_collision_free(env, q_near[0], q_near[1], q_new[0], q_new[1]):
            nodes.append(q_new); new_idx = len(nodes) - 1; parents[new_idx] = nearest_idx
            if math.hypot(q_new[0] - x_goal, q_new[1] - y_goal) < step_size and is_segment_collision_free(env, q_new[0], q_new[1], x_goal, y_goal):
                nodes.append((x_goal, y_goal)); parents[len(nodes) - 1] = new_idx
                path, curr = [], len(nodes) - 1
                while curr is not None: path.append(nodes[curr]); curr = parents[curr]
                return path[::-1]
    return None

q = env.random_query()
if q is not None:
    pl.ion()
    pl.figure(1)
    pl.clf()
    env.plot()
    env.plot_query(q[0], q[1], q[2], q[3])
    prm_path = solve_prm(env, q)
    if prm_path: pl.plot(np.array(prm_path)[:,0], np.array(prm_path)[:,1], 'g.-', linewidth=2, label='PRM Path')
    rrt_path = solve_rrt(env, q)
    if rrt_path: pl.plot(np.array(rrt_path)[:,0], np.array(rrt_path)[:,1], 'b.-', linewidth=2, label='RRT Path')
    pl.title("PRM (Green) vs RRT (Blue)")
    pl.legend()
    pl.show(block=True)
