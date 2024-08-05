import numpy as np
from scipy import stats
from sklearn.neighbors import NearestNeighbors

# Triangular Membership Function class
class TriangularMembershipFunction:
    def __init__(self, a, b, c):
        assert a <= b and b <= c, 'a, b and c must not be equal and must be in increasing order.'
        self.a = a
        self.b = b
        self.c = c
        self.slope = 1/(b-a)

    def fuzzify(self, x):
        if x < self.a or x > self.c:
            return 0
        if x < self.b:
            return self.y_left(x)
        else:
            return self.y_right(x)
        
    def y_left(self, x):
        y = self.slope * (x - self.b) + 1
        return y

    def y_right(self, x):
        y = -self.slope * (x - self.b) + 1
        return y
    
    def alpha_cut(self, y):
        assert 0 <= y <= 1, 'degree of truth must be between 0 and 1.'
        x_left = self.b - ((y - 1) / self.slope)
        x_right = self.b + ((y - 1) / self.slope)
        return x_left, x_right

# Membership functions
empty = TriangularMembershipFunction(-0.25, 0.00, 0.25)
sparse = TriangularMembershipFunction(0.00, 0.25, 0.50)
uniform = TriangularMembershipFunction(0.25, 0.50, 0.75)
dense = TriangularMembershipFunction(0.50, 0.75, 1.00)
full = TriangularMembershipFunction(0.75, 1.00, 1.25)

very_close = TriangularMembershipFunction(-0.25, 0.00, 0.25)
close = TriangularMembershipFunction(0.00, 0.25, 0.50)
halfway = TriangularMembershipFunction(0.25, 0.50, 0.75)
far = TriangularMembershipFunction(0.50, 0.75, 1.00)
very_far = TriangularMembershipFunction(0.75, 1.00, 1.25)

superfluous = TriangularMembershipFunction(0.00, 0.25, 0.50)
important = TriangularMembershipFunction(0.25, 0.50, 0.75)
essential = TriangularMembershipFunction(0.50, 0.75, 1.00)

def remove_outliers(points, n_neighbors=5, std_ratio=2.0):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(points)
    distances, _ = nbrs.kneighbors(points)
    
    mean_distances = np.mean(distances, axis=1)
    
    distance_std = np.std(mean_distances)
    distance_mean = np.mean(mean_distances)
    threshold = distance_mean + std_ratio * distance_std
    
    mask = mean_distances <= threshold
    return points[mask]

def fuzzy_downsample(input_cloud, importance_threshold):
    kde = stats.gaussian_kde(input_cloud.T)
    input_density = kde(input_cloud.T)
    
    distances = np.linalg.norm(input_cloud - np.mean(input_cloud, axis=0), axis=1)
    
    x_density = (input_density - np.min(input_density)) / (np.max(input_density) - np.min(input_density))
    x_distance = (distances - np.min(distances)) / (np.max(distances) - np.min(distances))
    
    defuzzified_importance = np.zeros(len(input_cloud))
    
    for i in range(len(input_cloud)):
        x_density_i = x_density[i]
        x_distance_i = x_distance[i]
        
        superfluous_firing_strength = 0.00
        important_firing_strength = 0.00
        essential_firing_strength = 0.00

        # Apply fuzzy rules
        if empty.fuzzify(x_density_i) > 0 and (very_close.fuzzify(x_distance_i) > 0 or close.fuzzify(x_distance_i) > 0 or halfway.fuzzify(x_distance_i) > 0):
            distance_membership = max(very_close.fuzzify(x_distance_i), close.fuzzify(x_distance_i), halfway.fuzzify(x_distance_i))
            firing_strength = min(empty.fuzzify(x_density_i), distance_membership)
            essential_firing_strength = firing_strength

        elif empty.fuzzify(x_density_i) > 0 and (far.fuzzify(x_distance_i) > 0 or very_far.fuzzify(x_distance_i) > 0):
            distance_membership = max(far.fuzzify(x_distance_i), very_far.fuzzify(x_distance_i))
            firing_strength = min(empty.fuzzify(x_density_i), distance_membership)
            superfluous_firing_strength = firing_strength

        if sparse.fuzzify(x_density_i) > 0 or uniform.fuzzify(x_density_i) > 0:
            density_membership = max(sparse.fuzzify(x_density_i), uniform.fuzzify(x_density_i))
            
            if very_close.fuzzify(x_distance_i) > 0 or close.fuzzify(x_distance_i) > 0:
                distance_membership = max(very_close.fuzzify(x_distance_i), close.fuzzify(x_distance_i))
                firing_strength = min(density_membership, distance_membership)
                essential_firing_strength = max(essential_firing_strength, firing_strength)
            
            if halfway.fuzzify(x_distance_i) > 0:
                firing_strength = min(density_membership, halfway.fuzzify(x_distance_i))
                important_firing_strength = max(important_firing_strength, firing_strength)
            
            if far.fuzzify(x_distance_i) > 0 or very_far.fuzzify(x_distance_i) > 0:
                distance_membership = max(far.fuzzify(x_distance_i), very_far.fuzzify(x_distance_i))
                firing_strength = min(density_membership, distance_membership)
                superfluous_firing_strength = max(superfluous_firing_strength, firing_strength)

        if dense.fuzzify(x_density_i) > 0:
            density_membership = dense.fuzzify(x_density_i)
            
            if very_close.fuzzify(x_distance_i) > 0 or close.fuzzify(x_distance_i) > 0:
                distance_membership = max(very_close.fuzzify(x_distance_i), close.fuzzify(x_distance_i))
                firing_strength = min(density_membership, distance_membership)
                important_firing_strength = max(important_firing_strength, firing_strength)
            else:
                firing_strength = density_membership
                superfluous_firing_strength = max(superfluous_firing_strength, firing_strength)

        if full.fuzzify(x_density_i) > 0 or far.fuzzify(x_distance_i) > 0 or very_far.fuzzify(x_distance_i) > 0:
            firing_strength = max(full.fuzzify(x_density_i), far.fuzzify(x_distance_i), very_far.fuzzify(x_distance_i))
            superfluous_firing_strength = max(superfluous_firing_strength, firing_strength)

        # Defuzzification
        points = np.zeros([8, 2])
        points[0] = (0.00, 0.00)
        points[1] = (0.15, superfluous_firing_strength)
        points[2] = (0.30, max(superfluous_firing_strength, important_firing_strength))
        points[3] = (0.45, max(superfluous_firing_strength, important_firing_strength))
        points[4] = (0.60, max(important_firing_strength, essential_firing_strength))
        points[5] = (0.60, max(important_firing_strength, essential_firing_strength))
        points[6] = (0.75, essential_firing_strength)
        points[7] = (1.00, 0.00)

        # Calculate center of gravity (CoA)
        defuzzified_importance[i] = (np.dot(points[:, 0], points[:, 1])) / np.sum(points[:, 1])
    
    # Downsample based on importance threshold
    downsampled_cloud = input_cloud[defuzzified_importance > importance_threshold]
    
    return downsampled_cloud


def process_array(array, shuffled_array, name, seed=42, n_neighbors=5, std_ratio=2.0, initial_importance_threshold=0.3, min_percentage=0.7, verbose=False):
    np.random.seed(seed)
    if verbose:
        print(f"\nProcessing {name}:")
        print(f"Original number of points: {len(array)}")
    cleaned_cloud = remove_outliers(array, n_neighbors=n_neighbors, std_ratio=std_ratio)
    if verbose:
        print(f"Number of points after outlier removal: {len(cleaned_cloud)}")
    
    # Step 2: Apply fuzzy downsampling with dynamic threshold adjustment
    importance_threshold = initial_importance_threshold
    final_downsampled = fuzzy_downsample(cleaned_cloud, importance_threshold)
    
    # while len(final_downsampled) < min_percentage * len(array) and importance_threshold > 0:
    #     importance_threshold -= 0.05
    #     final_downsampled = fuzzy_downsample(cleaned_cloud, importance_threshold)
    
    if verbose:
        print(f"Final number of points after fuzzy downsampling: {len(final_downsampled)}")
        print(f"Final importance threshold: {importance_threshold:.2f}")
        print(f"Percentage of original size: {len(final_downsampled) / len(array):.2%} \n")

    if shuffled_array is None:
        return final_downsampled
    
    # Adjust shuffled array to match the size of the downsampled array
    adjusted_shuffled = shuffled_array[:len(final_downsampled)]
    # print(f"Adjusted shuffled array size: {len(adjusted_shuffled)}")
    
    return final_downsampled, adjusted_shuffled