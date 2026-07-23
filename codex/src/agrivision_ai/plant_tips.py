import random

PLANT_TIPS = {

    "apple": [
        "Prune trees during dormancy to improve airflow.",
        "Remove fallen leaves to reduce fungal diseases.",
        "Water deeply instead of frequent shallow watering.",
        "Inspect leaves regularly for early disease symptoms."
    ],

    "cassava": [
        "Use healthy stem cuttings for planting.",
        "Keep fields free from weeds.",
        "Rotate crops to reduce disease buildup.",
        "Remove infected plants immediately."
    ],

    "cherry (including sour)": [
        "Prune branches to improve air circulation.",
        "Water at the base of the tree.",
        "Remove diseased leaves and fruits.",
        "Avoid overcrowding nearby plants."
    ],

    "corn (maize)": [
        "Maintain proper spacing between plants.",
        "Rotate crops every season.",
        "Monitor leaves for early disease signs.",
        "Keep weeds under control."
    ],

    "grape": [
        "Prune vines every year.",
        "Allow good sunlight inside the canopy.",
        "Avoid wetting leaves during irrigation.",
        "Remove infected grapes immediately."
    ],

    "orange": [
        "Inspect leaves regularly for pests.",
        "Remove diseased branches promptly.",
        "Maintain balanced fertilization.",
        "Water consistently during dry periods."
    ],

    "peach": [
        "Prune annually for healthy growth.",
        "Keep fallen leaves cleaned up.",
        "Water deeply once or twice a week.",
        "Inspect fruits frequently."
    ],

    "pepper, bell": [
        "Avoid overhead watering.",
        "Use mulch to retain soil moisture.",
        "Remove yellow leaves promptly.",
        "Provide adequate sunlight."
    ],

    "potato": [
        "Use certified disease-free seed potatoes.",
        "Avoid planting potatoes in the same field repeatedly.",
        "Keep soil well drained.",
        "Remove infected plants quickly."
    ],

    "rice": [
        "Maintain proper water levels.",
        "Use certified healthy seeds.",
        "Control weeds regularly.",
        "Monitor fields for insects and diseases."
    ],

    "squash": [
        "Provide plenty of sunlight.",
        "Improve airflow between plants.",
        "Avoid watering the leaves.",
        "Harvest fruits regularly."
    ],

    "strawberry": [
        "Remove old leaves after harvest.",
        "Keep fruits off wet soil.",
        "Water early in the morning.",
        "Use mulch to reduce fungal diseases."
    ],

    "tomato": [
        "Water the soil, not the leaves.",
        "Remove lower yellow leaves regularly.",
        "Stake plants for better airflow.",
        "Rotate crops every season."
    ]
}


def get_tip(plant_name):
    tips = PLANT_TIPS.get(
        plant_name.lower(),
        ["Inspect your plant regularly and maintain proper watering."]
    )
    return random.choice(tips)