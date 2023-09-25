import csv
import os
import random

def load_csv(filename):
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        return [row for row in reader]

def save_csv(filename, fieldnames, rows):
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def likert_evaluation(model_name, data):
    results = []
    for item in data:
        print(f"Evaluating hypothesis for bit: {item['bit']}")
        print(f"Flip: {item['flip']}\n")
        
        novelty = input("Rate the novelty on a scale of 1-5: ")
        creativity = input("Rate the creativity on a scale of 1-5: ")
        efficiency = input("Rate the efficiency on a scale of 1-5: ")
        practicality = input("Rate the practicality on a scale of 1-5: ")
        elegance = input("Rate the elegance on a scale of 1-5: ")

        results.append({
            'model': model_name,
            'bit': item['bit'],
            'flip': item['flip'],
            'novelty': novelty,
            'creativity': creativity,
            'efficiency': efficiency,
            'practicality': practicality,
            'elegance': elegance,
        })

        save_csv(f"likert_results_{model_name}.csv", ['model', 'bit', 'flip', 'novelty', 'creativity', 'efficiency', 'practicality', 'elegance'], results)

def comparison_evaluation(models, data_dict):
    results = []
    for _ in range(100):  # Number of comparisons
        model1, model2 = random.sample(models, 2)
        item1, item2 = random.choice(data_dict[model1]), random.choice(data_dict[model2])
        
        print(f"\nModel 1 ({model1}) Flip: {item1['flip']}")
        print(f"Model 2 ({model2}) Flip: {item2['flip']}")
        
        winner = input("Which model generated a better flip? Enter 1 for Model 1, 2 for Model 2: ")
        
        results.append({
            'model1': model1,
            'model2': model2,
            'bit1': item1['bit'],
            'bit2': item2['bit'],
            'flip1': item1['flip'],
            'flip2': item2['flip'],
            'winner': winner
        })

        save_csv("comparison_results.csv", ['model1', 'model2', 'bit1', 'bit2', 'flip1', 'flip2', 'winner'], results)

if __name__ == "__main__":
    mode = input("Enter the mode ('likert' or 'comparison'): ")
    
    if mode == 'likert':
        model_name = input("Enter the model name (e.g., gpt_4_zero_shot, hypogen): ")
        data = load_csv(f"{model_name}.csv")
        likert_evaluation(model_name, data)
    elif mode == 'comparison':
        models = ['gpt_4_zero_shot', 'gpt_4_three_shot', 'llama_finetuned_three_shot', 'llama_three_shot', 'hypogen']
        data_dict = {model: load_csv(f"{model}.csv") for model in models}
        comparison_evaluation(models, data_dict)
    else:
        print("Invalid mode. Please enter either 'likert' or 'comparison'.")