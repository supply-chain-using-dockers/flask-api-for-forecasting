
# coding: utf-8

# In[41]:

from flask import Flask, render_template, request, jsonify
from flask import json
import pandas as pd
import random
import codecs, json 



# In[42]:
app = Flask(__name__)


x = [1,0]
column2 = []
column1 = random.choices(x,k=156)
for i in range(0,156):
    column2.append(random.randint(0,100))
weather = ['sunny','rainy','stormy','windy','dusty','humid','foggy']
column3 = random.choices(weather,k=156)
natural_dis = ['earthquake', 'tsunami','landslide','cloudburst','floods','Nothing']
traffic = ['low','medium','high','no']
column4 = random.choices(natural_dis,k=156)
column5 = random.choices(x,k=156)
column6 = random.choices(x,k=156)
column7 = random.choices(traffic,k=156)
column8 = random.choices(x,k=156)
column9 = random.choices(x,k=156)
column10 = random.choices(x,k=156)


# In[43]:


df  = pd.DataFrame({'machine_breakdown':column1,'workmen_absentism':column2, 'weather':column3,'natural_disaster':column4,'order_bottleneck':column5, 'vehicle_breakdown':column6,'traffic':column7,'software_issues':column8,'bad_debts':column9,'unpredictable_delay':column10})
#df


# In[44]:


            
days = []
for i in range(0,156):
    if column1[i]==1 and (column7[i]=='high' or column7[i]=='medium') and column4[i]!='Nothing':
        days.append(random.randint(70,80))
    elif column1[i]==0 and column2[i]>40 and (column7[i]=='high' or column7[i]=='medium') and column4[i]!='Nothing':
        days.append(random.randint(60,70))
    elif column2[i]>40 and column1[i]==1 and ( column7[i]=='high' or column7[i]=='medium'):
        days.append(random.randint(30,50))
    else:
        days.append(random.randint(5,20))


# In[45]:


df = pd.get_dummies(df, columns=['weather','natural_disaster','traffic'])
#df


# In[46]:


df['no_of_days'] = days


# In[47]:


#list(df)


# In[48]:


x = df.iloc[:,0:24].values
y = df.iloc[:,24:].values


# In[49]:




from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()

x=sc_x.fit_transform(x)
#print(pd.DataFrame(x))
y=sc_y.fit_transform(y)
# print(pd.DataFrame(y))


# In[50]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3, random_state = 0)


# In[51]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor



# Fit regression model
# regr_1 = DecisionTreeRegressor(max_depth=500)

regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=500),
                          n_estimators=300, random_state=0)

# regr_1.fit(x_train, y_train)
regr_2.fit(x_train, y_train)

# y_pred1 = regr_1.predict(x_test)
y_pred2 = regr_2.predict(x_test)
from sklearn.metrics import r2_score
#r2_score(y_test, y_pred2)  


# In[53]:


@app.route('/whatup', methods=['POST','GET'])
def whatup():
    # first_name = request.form['first_name']
    # last_name = request.form['last_name']
    '''
    machine_breakdown = request.get_json['machine_breakdown']
    workmen_absentism =  request.get_json['workmen_absentism']
    order_bottleneck =  request.get_json['order_bottleneck']
    vehicle_breakdown = request.get_json['vehicle_breakdown']
    software_issues =  request.get_json['software_issues']
    bad_debts =  request.get_json['bad_debts']
    unpredictable_delay = get_request.json['unpredictable_delay']


    weather =  request.get_json['weather']
    natural_disaster =  request.get_json['natural_disaster']
    traffic =  request.get_json['traffic']
    '''
    data = request.get_json(force=True)
    machine_breakdown = data["machine_breakdown"]
    workmen_absentism =  data["workmen_absentism"]
    order_bottleneck =  data["order_bottleneck"]
    vehicle_breakdown = data["vehicle_breakdown"]
    software_issues =  data["software_issues"]
    bad_debts =  data["bad_debts"]
    unpredictable_delay = data["unpredictable_delay"]
    weather = data["weather"]
    natural_disaster = data["natural_disaster"]
    traffic = data["traffic"]
    
    if (weather =='dusty'):
        x1 = 1
    else:
        x1 = 0


    if (weather =='foggy'):
        x2 = 1
    else:
        x2 = 0

    if (weather =='humid'):
        x3 = 1
    else:
        x3 = 0

    if (weather =='rainy'):
        x4 = 1
    else:
        x4 = 0

    if (weather =='stormy'):
        x5 = 1
    else:
        x5 = 0

    if (weather =='sunny'):
        x6 = 1
    else:
        x6 = 0

    if (weather =='windy'):
        x7 = 1
    else:
        x7 = 0


    if (natural_disaster =='Nothing'):
        x8 = 1
    else:
        x8 = 0

    if (natural_disaster =='cloudburst'):
        x9 = 1
    else:
        x9 = 0

    if (natural_disaster =='earthquake'):
        x10 = 1
    else:
        x10 = 0

    if (natural_disaster =='floods'):
        x11 = 1
    else:
        x11 = 0

    if (natural_disaster =='landslide'):
        x12 = 1
    else:
        x12 = 0


    if (natural_disaster =='tsunami'):
        x13 = 1
    else:
        x13 = 0



    if (traffic =='high'):
        x14 = 1
    else:
        x14 = 0
  
    if (traffic =='low'):
        x15 = 1
    else:
        x15 = 0

    if (traffic =='medium'):
        x16 = 1
    else:
        x16 = 0

    if (traffic =='no'):
        x17 = 1
    else:
        x17 = 0





    import numpy as np
    result = sc_y.inverse_transform(
            regr_2.predict(
                sc_x.transform(
                    np.array([[int(machine_breakdown),int(workmen_absentism),int(order_bottleneck),int(vehicle_breakdown),int(software_issues),int(bad_debts),int(unpredictable_delay),int(x1),int(x2),int(x3),int(x4),int(x5),int(x6),int(x7),int(x8),int(x9),int(x10),int(x11),int(x12),int(x13),int(x14),int(x15),int(x16),int(x17)]])
                )
            )
        )
    

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    json_dump = json.dumps({'le bsdk PALAK le': result}, cls=NumpyEncoder)




    return json_dump
    #return jsonify({'result': "horaha hai" })
    
if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 4200, debug = True)