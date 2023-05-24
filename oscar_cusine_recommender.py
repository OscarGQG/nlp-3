#!/usr/bin/env python
# coding: utf-8

# In[1]:


#1 Load the data and carry out some basic data analysis and exploration
import json

with open('recipies.json', 'r') as f:
    data = json.load(f)


# In[2]:


num_recipes = len(data)
print("Total number of instances of recipes:", num_recipes)


# In[3]:


cuisines = set()
for recipe in data:
    cuisines.add(recipe['cuisine'])
num_cuisines = len(cuisines)
print("Number of cuisines available in the data:", num_cuisines)


# In[4]:


cuisine_recipe_counts = {}
for recipe in data:
    cuisine = recipe['cuisine']
    if cuisine in cuisine_recipe_counts:
        cuisine_recipe_counts[cuisine] += 1
    else:
        cuisine_recipe_counts[cuisine] = 1


# In[5]:


print("Cuisine | \tNumber of Recipes")
for cuisine, count in cuisine_recipe_counts.items():
    print(cuisine + " | \t" + str(count))


# In[6]:


from apyori import apriori

ingredients = []
for recipe in data:
    if recipe['cuisine'] == cuisine:
        ingredients.append(recipe['ingredients'])


# In[18]:


while True:
    cuisine_type = input('\nEnter a cuisine type (or "exit" to quit): ').lower()
    if cuisine_type == 'exit':
        break
    if cuisine_type not in cuisines:
        print(f'Sorry, we don\'t have recommendations for {cuisine_type}. Please try a different cuisine type.')
        continue
    
    # Analyze the ingredients using the apriori algorithm
    cuisine_data = [recipe['ingredients'] for recipe in data if recipe['cuisine'] == cuisine_type]
    recipes = [r for r in data if r['cuisine'] == cuisine_type]    
    support = 100 * len(cuisine_data) // num_recipes
    rules = apriori(cuisine_data, min_support=support/100, min_confidence=0.5, min_lift=2)
        
    # Get the top group of ingredients and rules with lift > 2
    results = list(rules)
    top_ingredients = results[0].items
    lift_rules = [r for r in results if r.ordered_statistics[0].lift > 2]
        
    # Print the recommendations
    print(f'Top ingredients for {cuisine_type}:')
    for ingredient in top_ingredients:
        print(f'- {ingredient}')
    print('\nRules with lift > 2:')
    for rule in lift_rules:
        lhs = ', '.join(list(rule.ordered_statistics[0].items_base))
        rhs = ', '.join(list(rule.ordered_statistics[0].items_add))
        lift_value = rule.ordered_statistics[0].lift
        print(f'- {lhs} -> {rhs} (lift = {lift_value:.2f})')        


# In[ ]:




