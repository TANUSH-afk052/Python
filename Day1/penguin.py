import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load dataset as came with seaborn
penguins = sns.load_dataset("penguins")

# Set theme
sns.set_style("whitegrid")
sns.set_palette("coolwarm")

# Basic info
print(penguins.head())
print(penguins.info())

# 1. Histogram of body mass
plt.figure(figsize=(6, 4))
sns.histplot(data=penguins, x="body_mass_g", hue="species", multiple="stack", kde=True)
plt.title("Body Mass by Species")
plt.show()

# 2. Count plot: Count species by island
plt.figure(figsize=(6, 4))
sns.countplot(data=penguins, x="island", hue="species")
plt.title("Penguin Species per Island")
plt.show()

# 3. Bar plot: Average body mass by species and sex
plt.figure(figsize=(6, 4))
sns.barplot(data=penguins, x="species", y="body_mass_g", hue="sex", estimator=np.mean)
plt.title("Average Body Mass by Species and Sex")
plt.show()

# 4. Box plot: Flipper length distribution by species and sex
plt.figure(figsize=(6, 4))
sns.boxplot(data=penguins, x="species", y="flipper_length_mm", hue="sex")
plt.title("Flipper Length by Species and Sex")
plt.show()

# 5. Scatter plot: Bill length vs bill depth colored by species
plt.figure(figsize=(6, 4))
sns.scatterplot(data=penguins, x="bill_length_mm", y="bill_depth_mm", hue="species", style="sex")
plt.title("Bill Length vs Depth by Species")
plt.show()

# 6. Joint plot: distribution + relationship in a single view
sns.jointplot(data=penguins, x="bill_length_mm", y="bill_depth_mm", hue="species", kind="kde")
plt.suptitle("Joint Distribution of Bill Length & Depth", y=1.02)
plt.show()

# 7. Pair plot: all pairwise numerical relationships
sns.pairplot(data=penguins, hue="species")
plt.suptitle("Pairplot of Penguin Measurements", y=1.02)
plt.show()

# 8. FacetGrid: separate plots for each species and sex
g = sns.FacetGrid(penguins, col="species", row="sex")
g.map(sns.scatterplot, "bill_length_mm", "body_mass_g")
g.add_legend()
plt.suptitle("Body Mass vs Bill Length by Species & Sex", y=1.03)
plt.show()
