import matplotlib.pyplot as plt

# Data
data = [
    {'name': 'IonQ Aria (Azure)', 'error': 18.147429897608692, 'time': 942.75, 'price': 4875.0},
    {'name': 'Quantinuum H1', 'error': 2.554021344441282, 'time': 15162.352941176474, 'price': 44750.00000000001},
    {'name': 'Quantinuum H2', 'error': 3.807334062755041, 'time': 15162.352941176474, 'price': 48330.00000000001},
    {'name': 'Rigetti Ankaa-9Q-3', 'error': 31.764464750740217, 'time': 0.19200000000000003,
     'price': 0.24960000000000004},
    {'name': 'ibm_kyiv', 'error': 38.192116311170246, 'time': 1.276852, 'price': 2.0429632},
    {'name': 'ibm_sherbrooke', 'error': 27.725692660115296, 'time': 1.2688670000000002, 'price': 2.0301872000000003},
    {'name': 'ibm_brisbane', 'error': 28.034665486706256, 'time': 1.509, 'price': 2.4143999999999997},
    {'name': 'IonQ Aria (Amazon)', 'error': 18.147429897608692, 'time': 942.75, 'price': 1515.0},
    {'name': 'IQM Garnet', 'error': 26.27917220330216, 'time': 0.18299999999999997, 'price': 87.5}
]

# Extract data for plotting
names = [item['name'] for item in data]
errors = [item['error'] for item in data]
times = [item['time'] for item in data]
prices = [item['price'] for item in data]


# Define a function to create a bar chart with improved layout
def plot_vertical_bar_chart(values, labels, ylabel, color, unit, filename):
    plt.figure(figsize=(14, 10))  # Increased figure size for better spacing
    bars = plt.bar(labels, values, color=color)

    # Label each bar with its value, adjust position and font size to avoid overlap
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.02,
            f"{value:.2f}{unit}",
            ha='center',
            va='bottom',
            color='black',
            fontsize=10  # Reduced font size to fit better
        )

    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha='right', fontsize=10)  # Adjust rotation and font size
    plt.subplots_adjust(bottom=0.2)  # Add space at the bottom to avoid overlap
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# Create individual bar charts with improved layout
plot_vertical_bar_chart(errors, names, 'Error (%)', 'lightcoral', '%', 'error_histogram.png')
plot_vertical_bar_chart(times, names, 'Time (s)', 'lightblue', ' s', 'time_histogram.png')
plot_vertical_bar_chart(prices, names, 'Price ($)', 'lightgreen', ' $', 'price_histogram.png')
