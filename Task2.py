#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Sample data
data = {
    'material_id': [101, 102, 103, 104, 105],
    'title': ["Math Basics", "Advanced Physics", "Chemistry 101", "History of Art", "Programming with Python"],
    'description': [
        "Basic concepts of mathematics.",
        "Advanced topics in physics.",
        "Introduction to Chemistry.",
        "Overview of art history.",
        "Learning programming using Python."
    ]
}

# Convert to DataFrame
materials_df = pd.DataFrame(data)

# Content-Based Filtering using TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(materials_df['description'])

# Function to get content-based recommendations
def get_content_based_recommendations(material_id, tfidf_matrix, materials_df):
    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
    sim_scores = list(enumerate(cosine_similarities[material_id]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Get top 5 similar materials
    material_indices = [i[0] for i in sim_scores]
    return materials_df.iloc[material_indices]

# Example usage
material_id = 0  # Assuming you want recommendations for the first material
recommended_materials = get_content_based_recommendations(material_id, tfidf_matrix, materials_df)
print(recommended_materials)


# In[ ]:




