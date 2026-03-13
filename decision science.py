import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# STEP 1 — PROVIDERS
# -----------------------------

providers = [f"3PL-{i}" for i in range(1,21)]

criteria = ["Price","Delivery","Safety","Technology","Social"]

# -----------------------------
# STEP 2 — INPUT DATA
# -----------------------------

data = {

"Price":[0.2240,0.2263,0.2216,0.2240,0.2193,0.2287,0.2145,0.2240,0.2310,0.2287,
         0.2145,0.2169,0.2263,0.2240,0.2169,0.2287,0.2216,0.2216,0.2310,0.2263],

"Delivery":[0.2286,0.2055,0.2276,0.2272,0.2283,0.2278,0.2262,0.2263,0.2281,0.2269,
            0.2267,0.2263,0.2032,0.2240,0.2279,0.2254,0.2056,0.2286,0.2231,0.2249],

"Safety":[0.2017,0.2017,0.2269,0.2522,0.2522,0.2017,0.2522,0.2522,0.1513,0.2269,
          0.2017,0.1765,0.2269,0.2522,0.2269,0.2269,0.2269,0.2017,0.2522,0.2269],

"Technology":[0.2594,0.2335,0.2335,0.2335,0.2335,0.2075,0.2075,0.2335,0.2594,0.2335,
              0.2594,0.2075,0.1816,0.1556,0.2594,0.2075,0.2335,0.2075,0.2075,0.1816],

"Social":[0.2191,0.2434,0.2434,0.2191,0.1947,0.2191,0.2434,0.1947,0.1947,0.2191,
          0.2434,0.1947,0.2191,0.2434,0.2434,0.2434,0.2191,0.1947,0.2191,0.2434]
}

df = pd.DataFrame(data,index=providers)

print("\n===== INPUT DATA =====")
print(df)

# =====================================
# STEP 3 — FUZZY AHP WEIGHTS
# =====================================

fuzzy_matrix = np.array([

[(1,1,1),(2,2.5,3),(2.5,3,3.5),(9,10,11),(15,16,17)],

[(1/3,1/2.5,1/2),(1,1,1),(1,1.1,1.2),(4,4.5,5),(6,7,8)],

[(1/3.5,1/3,1/2.5),(1/1.2,1/1.1,1),(1,1,1),(3.5,4,4.5),(5.5,6,6.5)],

[(1/11,1/10,1/9),(1/5,1/4.5,1/4),(1/4.5,1/4,1/3.5),(1,1,1),(1.4,1.5,1.6)],

[(1/17,1/16,1/15),(1/8,1/7,1/6),(1/6.5,1/6,1/5.5),(1/1.6,1/1.5,1/1.4),(1,1,1)]

])
n=len(criteria)

geo_means=[]

for i in range(n):

    l,m,u=1,1,1

    for j in range(n):

        l*=fuzzy_matrix[i][j][0]
        m*=fuzzy_matrix[i][j][1]
        u*=fuzzy_matrix[i][j][2]

    geo_means.append((l**(1/n),m**(1/n),u**(1/n)))

geo_means=np.array(geo_means)

sum_l=np.sum(geo_means[:,0])
sum_m=np.sum(geo_means[:,1])
sum_u=np.sum(geo_means[:,2])

fuzzy_weights=[]

for i in range(n):

    l=geo_means[i][0]/sum_u
    m=geo_means[i][1]/sum_m
    u=geo_means[i][2]/sum_l

    fuzzy_weights.append((l,m,u))

weights=[]

for w in fuzzy_weights:
    weights.append((w[0]+w[1]+w[2])/3)

weights=np.array(weights)
weights=weights/np.sum(weights)

print("\n===== FUZZY AHP WEIGHTS =====")

for c,w in zip(criteria,weights):
    print(c,":",round(w,4))

# =====================================
# STEP 4 — TOPSIS
# =====================================

weighted = df * weights

ideal = weighted.max()
anti = weighted.min()

S_plus = np.sqrt(((weighted - ideal)**2).sum(axis=1))
S_minus = np.sqrt(((weighted - anti)**2).sum(axis=1))

C = S_minus/(S_plus+S_minus)

topsis=pd.DataFrame({"TOPSIS Score":C})
topsis["TOPSIS Rank"]=topsis["TOPSIS Score"].rank(ascending=False)

# =====================================
# STEP 5 — FUZZY INFERENCE SYSTEM
# =====================================

price = ctrl.Antecedent(np.arange(0,1.01,0.01),'price')
delivery = ctrl.Antecedent(np.arange(0,1.01,0.01),'delivery')
safety = ctrl.Antecedent(np.arange(0,1.01,0.01),'safety')
tech = ctrl.Antecedent(np.arange(0,1.01,0.01),'tech')
social = ctrl.Antecedent(np.arange(0,1.01,0.01),'social')

score = ctrl.Consequent(np.arange(0,1.01,0.01),'score')

for var in [price,delivery,safety,tech,social]:

    var['low']=fuzz.trimf(var.universe,[0,0,0.5])
    var['medium']=fuzz.trimf(var.universe,[0,0.5,1])
    var['high']=fuzz.trimf(var.universe,[0.5,1,1])

score['poor']=fuzz.trimf(score.universe,[0,0,0.5])
score['average']=fuzz.trimf(score.universe,[0,0.5,1])
score['good']=fuzz.trimf(score.universe,[0.5,1,1])

rules=[]
levels=['low','medium','high']

for p in levels:
    for d in levels:
        for s in levels:

            if p=='low' and d=='high' and s=='high':
                rules.append(ctrl.Rule(price[p] & delivery[d] & safety[s],score['good']))

            elif p=='medium' and d=='medium' and s=='medium':
                rules.append(ctrl.Rule(price[p] & delivery[d] & safety[s],score['average']))

            else:
                rules.append(ctrl.Rule(price[p] & delivery[d] & safety[s],score['poor']))

rules.append(ctrl.Rule(tech['high'] & social['high'],score['good']))
rules.append(ctrl.Rule(tech['low'] | social['low'],score['poor']))

system=ctrl.ControlSystem(rules)
sim=ctrl.ControlSystemSimulation(system)

fis_scores=[]

for i,row in df.iterrows():

    sim.input['price']=row["Price"]
    sim.input['delivery']=row["Delivery"]
    sim.input['safety']=row["Safety"]
    sim.input['tech']=row["Technology"]
    sim.input['social']=row["Social"]

    sim.compute()

    fis_scores.append(sim.output['score'])

fis=pd.DataFrame({"FIS Score":fis_scores},index=providers)
fis["FIS Rank"]=fis["FIS Score"].rank(ascending=False)

# =====================================
# STEP 6 — FINAL COMPARISON
# =====================================

result=topsis.join(fis)

print("\n===== FINAL COMPARISON =====")
print(result.sort_values("TOPSIS Rank"))

# =====================================
# VISUAL 1 — TOPSIS SCORES
# =====================================

plt.figure()
plt.bar(providers,result["TOPSIS Score"])
plt.xticks(rotation=90)
plt.title("TOPSIS Scores of 3PL Providers")
plt.show()

# =====================================
# VISUAL 2 — FIS SCORES
# =====================================

plt.figure()
plt.bar(providers,result["FIS Score"])
plt.xticks(rotation=90)
plt.title("FIS Scores of 3PL Providers")
plt.show()

# =====================================
# VISUAL 3 — RANK COMPARISON
# =====================================

plt.figure()
plt.plot(providers,result["TOPSIS Rank"],marker='o',label="TOPSIS")
plt.plot(providers,result["FIS Rank"],marker='s',label="FIS")
plt.legend()
plt.xticks(rotation=90)
plt.title("Ranking Comparison")
plt.show()

# =====================================
# VISUAL 4 — SENSITIVITY ANALYSIS
# =====================================

price_weight_range=np.linspace(0.3,0.7,20)

scores=[]

for w in price_weight_range:

    new_weights=np.array([w,0.2111,0.1951,0.0479,0.0311])

    weighted=df*new_weights

    ideal=weighted.max()
    anti=weighted.min()

    S_plus=np.sqrt(((weighted-ideal)**2).sum(axis=1))
    S_minus=np.sqrt(((weighted-anti)**2).sum(axis=1))

    C=S_minus/(S_plus+S_minus)

    scores.append(C.max())

plt.figure()
plt.plot(price_weight_range,scores)
plt.title("Sensitivity Analysis: Price Weight Effect")
plt.xlabel("Price Weight")
plt.ylabel("Best TOPSIS Score")
plt.show()

# =====================================
# VISUAL 5 — CORRELATION HEATMAP
# =====================================

analysis_df=df.copy()
analysis_df["TOPSIS Score"]=result["TOPSIS Score"]

corr_matrix=analysis_df.corr()

plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix,annot=True,cmap="coolwarm")
plt.title("Criteria Correlation Heatmap")
plt.show()

# =====================================
# VISUAL 6 — CRITERIA INFLUENCE
# =====================================

criteria_importance=corr_matrix["TOPSIS Score"].drop("TOPSIS Score")

plt.figure()
criteria_importance.sort_values().plot(kind="barh")
plt.title("Influence of Criteria on Provider Ranking")
plt.xlabel("Correlation with TOPSIS Score")
plt.show()

# =====================================
# VISUAL 7 — MONTE CARLO ROBUSTNESS
# =====================================

iterations=5000

rank1_count={p:0 for p in providers}

for _ in range(iterations):

    rand_weights=np.random.rand(5)
    rand_weights=rand_weights/rand_weights.sum()

    weighted=df*rand_weights

    ideal=weighted.max()
    anti=weighted.min()

    S_plus=np.sqrt(((weighted-ideal)**2).sum(axis=1))
    S_minus=np.sqrt(((weighted-anti)**2).sum(axis=1))

    C=S_minus/(S_plus+S_minus)

    best=C.idxmax()

    rank1_count[best]+=1

robustness=pd.DataFrame.from_dict(rank1_count,orient="index",columns=["Times Ranked #1"])

robustness["Probability"]=robustness["Times Ranked #1"]/iterations

robustness=robustness.sort_values("Times Ranked #1",ascending=False)

print("\n===== MONTE CARLO ROBUSTNESS TEST =====")
print(robustness)

top5=robustness.head(5)

plt.figure()
top5["Probability"].plot(kind="bar")
plt.title("Top Providers by Robustness (Monte Carlo Simulation)")
plt.ylabel("Probability of Being Ranked #1")
plt.show()