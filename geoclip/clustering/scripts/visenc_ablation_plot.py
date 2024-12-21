import matplotlib.pyplot as plt
import numpy as np

big_losses = [
    3.95816,
    3.65877,
    3.61815,
    3.38896,
    3.24973,
    3.16374,
    3.19751,
    3.23258,
    3.08487,
    3.03053,
    3.03467,
    3.00955,
    3.01976,
    3.01943,
    2.71415,
]

large_losses = [
    2.95103,
    2.74676,
    2.57477,
    2.57163,
    2.42464,
    2.45367,
    2.29884,
    2.39835,
    2.22231,
    2.18755,
    2.25719,
    2.06746,
    2.21403,
    2.07138,
    2.08222
]

siglip_losses = [
   2.77380,
   2.60472,
   2.33897,
   2.21343,
   2.19245,
   2.34324,
   1.90479,
   2.13256,
   2.05122,
   2.04308,
   1.80883,
   1.89991,
   1.79284,
   1.86316,
   1.82966
]



# Create the plot
plt.figure(figsize=(10, 6))

# Plot each loss list
plt.plot(big_losses, label='ViT-B-32')
plt.plot(large_losses, label='ViT-L-14')
plt.plot(siglip_losses, label='ViT-SO400M-14-SigLIP-384')

# Add title and labels
plt.title('Training losses for Visual Encoder ablations')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Add legend
plt.legend()

# Show grid for better readability
plt.grid(True)

# Display the plot
plt.show()


prediction_resolutions_km = [2500,750,200,25,1]
big_final_accuracy = [0.6040219907407407,
 0.3863329475308642,
 0.21207079475308643,
 0.12437307098765432,
 0.056158371913580245]

large_final_accuracy = [0.7555941358024691, 
0.5697579089506173, 
0.3584104938271605, 
0.2175684799382716, 
0.09010898919753087, 
]

siglip_final_accuracy = [ 0.7771990740740741, 
0.6085310570987654, 
0.4090470679012346, 
0.25265239197530864,
0.10515528549382716
]


plt.figure(figsize=(10, 6))

# Set the x-axis positions for the bars
x = np.arange(len(prediction_resolutions_km))
bar_width = 0.2


# Plot the bars for each model
plt.bar(x - bar_width, big_final_accuracy, bar_width, label='ViT-B-32')
plt.bar(x, large_final_accuracy, bar_width, label='ViT-L-14')
plt.bar(x + bar_width, siglip_final_accuracy, bar_width, label='ViT-SO400M-14-SigLIP-384')

# Set the x-axis ticks and labels
plt.xticks(x, prediction_resolutions_km)
plt.xlabel('Spatial Resolution (km)')
plt.ylabel('Accuracy')
plt.title('Model Accuracies at Different Spatial Resolutions')

# Add legend
plt.legend()

# Show grid for better readability
plt.grid(axis='y')

# Display the plot
plt.show()