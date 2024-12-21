import matplotlib.pyplot as plt
import numpy as np

b128_losses = [1.67837,
               1.70005,
               1.57408,
               1.60811,
               1.35129,
               1.41907,
               1.46102,
               1.28730,
               1.08572,
               1.35926
]

b256_losses = [1.81209,
               1.92561,
               1.78649,
               1.84181,
               1.80979,
               1.95448,
               1.77811,
               1.59778,
               1.59529,
               1.46247
]

b512_losses = [2.44782,
               2.34880,
               2.37190,
               2.34761,
               2.32442,
               2.20644,
               2.10225,
               2.11637,
               1.95455,
               1.93979
]

b1024_losses = [2.99924,
                2.93855,
                2.86484,
                2.88525,
                2.64066,
                2.67655,
                2.53984,
                2.58635,
                2.63900,
                2.51418

]

b2048_losses = [3.56604,
                3.50898,
                3.45479,
                3.34281,
                3.25762,
                3.32278,
                3.19222,
                3.10914,
                3.10445,
                3.07099
]



# Create the plot
plt.figure(figsize=(10, 6))

# Plot each loss list
plt.plot(b128_losses, label='Batch size:128')
plt.plot(b256_losses, label='Batch size:256')
plt.plot(b512_losses, label='Batch size:512')
plt.plot(b1024_losses, label='Batch size:1024')
plt.plot(b2048_losses, label='Batch size:2048')

# Add title and labels
plt.title('Training losses for different batch sizes')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Add legend
plt.legend()

# Show grid for better readability
plt.grid(True)

# Display the plot
plt.show()


prediction_resolutions_km = [2500,750,200,25,1]
b128_final_accuracy = [ 0.7501687885802469, 
0.5611255787037037, 
0.34661940586419754, 
0.20105131172839505, 
0.08232060185185185
]

b256_final_accuracy = [0.753930362654321,
0.5678771219135802, 
0.35614390432098764, 
0.21257716049382716, 
0.08924093364197531, 
]

b512_final_accuracy = [0.7507716049382716, 
0.5692033179012346, 
0.3647038966049383, 
0.22159529320987653, 
0.09661940586419752, 
]

b1024_final_accuracy = [0.75390625, 
0.5706787109375, 
0.365625, 
0.2282958984375, 
0.10185546875, 
]

b2048_final_accuracy = [ 0.7541748046875, 
0.5749267578125, 
0.3677001953125, 
0.2298095703125, 
0.1055419921875, 
]


plt.figure(figsize=(10, 6))

# Set the x-axis positions for the bars
x = np.arange(len(prediction_resolutions_km))
bar_width = 0.1


# Plot the bars for each model
plt.bar(x - 2*bar_width, b128_final_accuracy, bar_width, label='Batch size:128')
plt.bar(x - bar_width, b256_final_accuracy, bar_width, label='Batch size:256')
plt.bar(x, b512_final_accuracy, bar_width, label='Batch size:512')
plt.bar(x + bar_width, b1024_final_accuracy, bar_width, label='Batch size:1024')
plt.bar(x + 2*bar_width, b2048_final_accuracy, bar_width, label='Batch size:2048')

# Set the x-axis ticks and labels
plt.xticks(x, prediction_resolutions_km)
plt.xlabel('Spatial Resolution (km)')
plt.ylabel('Accuracy')
plt.title('Accuracies at Different Spatial Resolutions for given batch-sizes')

# Add legend
plt.legend()

# Show grid for better readability
plt.grid(axis='y')

# Display the plot
plt.show()