import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds

# Sample Data (User-Item Ratings Matrix)
data = {
    'UserID': [1, 1, 1, 2, 2, 2, 3, 3, 4, 4],
    'ItemID': [101, 102, 103, 101, 103, 104, 102, 103, 101, 104],
    'Rating': [5, 3, 4, 4, 5, 3, 2, 4, 4, 5]
}
df = pd.DataFrame(data)

# Create User-Item Matrix
user_item_matrix = df.pivot(index='UserID', columns='ItemID', values='Rating').fillna(0)

# ---- User-Based Collaborative Filtering ----
user_similarity = cosine_similarity(user_item_matrix)
user_sim_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

def recommend_user_based(user_id, num_recommendations=2):
    similar_users = user_sim_df[user_id].sort_values(ascending=False).index[1:num_recommendations+1]
    recommendations = df[df['UserID'].isin(similar_users)].groupby('ItemID')['Rating'].mean().sort_values(ascending=False).index.tolist()
    return recommendations[:num_recommendations]

# ---- Item-Based Collaborative Filtering ----
item_similarity = cosine_similarity(user_item_matrix.T)
item_sim_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)

def recommend_item_based(item_id, num_recommendations=2):
    similar_items = item_sim_df[item_id].sort_values(ascending=False).index[1:num_recommendations+1]
    return similar_items.tolist()

# ---- Matrix Factorization (SVD) ----
# Convert user-item matrix to a NumPy array (convert to float for SVD compatibility)
user_item_matrix_np = user_item_matrix.to_numpy().astype(float)

# Perform Singular Value Decomposition (SVD)
U, sigma, Vt = svds(user_item_matrix_np, k=2)  # k = number of latent factors
sigma = np.diag(sigma)

# Reconstruct the matrix
reconstructed_matrix = np.dot(np.dot(U, sigma), Vt)

def recommend_svd(user_id, num_recommendations=2):
    user_index = user_id - 1  # Convert to 0-based index
    predictions = reconstructed_matrix[user_index]
    recommended_items = np.argsort(predictions)[::-1][:num_recommendations]
    return [user_item_matrix.columns[i] for i in recommended_items]

# Example Usage
print("User-Based Recommendations for User 1:", recommend_user_based(1))
print("Item-Based Recommendations for Item 101:", recommend_item_based(101))
print("SVD-Based Recommendations for User 1:", recommend_svd(1))
