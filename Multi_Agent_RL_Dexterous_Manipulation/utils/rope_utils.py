import cv2
import numpy as np
import heapq

sensitivity = 20
l_b=np.array([45,10,10])# lower hsv bound for green
u_b=np.array([85,255,255])# upper hsv bound to green
kernel = np.ones((31,31),np.uint8)

class RopeSearch:
    def __init__(self, rope_coms, skeleton):
        self.rope_coms = rope_coms
        self.inv_rope_coms = {com.tobytes(): n for n, com in enumerate(self.rope_coms)}
        self.skeleton = skeleton
        self.map = {}
        self.make_map()
        self.graph = {}
    
    def make_map(self):
        self.end_points = []
        for idx in range(len(self.rope_coms)):
            neighbors = self.get_neighbors(idx)
            self.map[idx] = neighbors
            if (len(neighbors) == 1) or (len(neighbors) >= 3):
                self.end_points.append(idx)
                
    def get_neighbors(self, idx):
        xy = self.rope_coms[idx]
        crop = np.transpose(np.where(self.skeleton[xy[0]-1:xy[0]+2, xy[1]-1:xy[1]+2]==1)) - 1
        locs = xy + crop
        neighbors = locs[np.transpose(np.where(np.logical_or(locs[:,0]!=xy[0],locs[:,1]!=xy[1])))]
        neighbors = [self.inv_rope_coms[nbr.tobytes()] for nbr in neighbors]
        return neighbors
    
    def dijkstra(self):
        for idx in self.end_points:
            distances = {node: float('inf') for node in self.map}
            distances[idx] = 0
            visited = set()
            queue = [(0, idx)]
            while queue:
                dist, node = heapq.heappop(queue)
                if node in visited:
                    continue
                visited.add(node)
                for neighbor in self.map[node]:
                    if neighbor in visited:
                        continue
                    dist += 1
                    if dist < distances[neighbor]:
                        distances[neighbor] = dist
                        self.graph[neighbor] = node
                        heapq.heappush(queue, (dist, neighbor))
                        
            for idx2 in self.end_points:
                if distances[idx2] > self.rope_coms.shape[0]*0.9:
                    path = [idx2]
                    while idx2 != idx:
                        idx2 = self.graph[idx2]
                        path.append(idx2)
                    path.append(idx)
                    return path
                
def get_rope_coms(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(frame, l_b, u_b)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours,_= cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    max_contour = contours[0]
    for contour in contours:
        if cv2.contourArea(contour)>cv2.contourArea(max_contour):
            max_contour=contour

    frame = cv2.resize(frame, (frame.shape[1]//3, frame.shape[0]//3))
    mask = cv2.resize(mask, (mask.shape[1]//3, mask.shape[0]//3))
    max_contour //= 3

#     kernel = np.ones((31,31),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    skeleton = skeletonize(mask//255, method='lee')
    rope_coms = np.array(np.where(skeleton==1)).T

    rope_search = RopeSearch(rope_coms, skeleton)
    rope_path = rope_search.dijkstra()
    rope_points = rope_coms[rope_path]

    frame = cv2.cvtColor(frame, cv2.COLOR_HSV2RGB)
    cv2.drawContours(frame, max_contour, -1, (0,255,0), 3)
    frame = cv2.polylines(frame, [np.flip(rope_points)], False, (0,255,255), 3)
    plt.figure(figsize=(16,9))
    plt.imshow(frame)
    return rope_points