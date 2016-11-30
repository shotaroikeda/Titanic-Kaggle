import pandas as pd
import numpy as np
import pylab as P
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


def convert_to_np(fname):
    df = pd.read_csv(fname, header=0)

    # Keep standard deviation ~1 percent
    df['Gender'] = df['Sex'].map({'male': -1, 'female': 0}).astype(int)
    # df['below_15'] = 0
    df.loc[(df.Age <= 15), 'Gender'] = 1

    # Find median ages
    median_ages = np.zeros((3, 3))
    for i in range(0,3):
        for j in range(0,3):
            median_ages[i][j] = df[(df['Gender'] == i-1) &
                                   (df['Pclass'] == j+1)]['Age'].dropna().median()

    df['AgeFill'] = df['Age']

    for i in range(0,3):
        for j in range(0,3):
            df.loc[(df.Age.isnull()) & (df.Gender == i-1) & (df.Pclass == j+1),
                   'AgeFill'] = median_ages[i][j]

    df['AgeIsNull'] = pd.isnull(df.Age).astype(int)

    # Extract gender and age features
    df['Gender*Age'] = (1 / df.AgeFill) * (df.Gender + 2) # Avoid 0

    df['FamilySize'] = df['SibSp'] + df['Parch']
    df['Age*Class'] = df.AgeFill * df.Pclass

    # Find median fares
    median_fare = np.zeros((3,3,10))
    for i in range(0,3):
        for j in range(0,3):
            for k in range(0,10):
                median_fare[i][j][k] = df[(df['Gender'] == i-1) &
                                          (df['Pclass'] == j+1) &
                                          (df['FamilySize'] == k)]['Fare'].dropna().median()

    df['FareFill'] = df['Fare']

    for i in range(0,3):
        for j in range(0,3):
            for k in range(0,10):
                df.loc[(df.Fare.isnull()) & (df['Gender'] == i-1) & (df['Pclass'] == j+1) &
                       (df['FamilySize'] == k), 'FareFill'] = median_fare[i][j][k]

    df['FareIsNull'] = pd.isnull(df.Fare).astype(int)

    # return df
    df = df.drop(['Fare', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age', 'Fare', 'PassengerId'], axis=1)
    df = df.dropna()
    return df.values

def age_group(df, min=0, max=1):
    return df[(df['AgeFill'] >= 0) & (df['AgeFill'] <= max)]

def extract(s_df, l_df):
    df = pd.DataFrame(index=np.arange(len(s_df)),
                      columns=l_df.columns.tolist())
    for i in range(len(s_df)):
        # print l_df[l_df.Name == s_df.Name.loc[i]]
        df.loc[i] = l_df[(l_df.Name == s_df.loc[i].Name)         &
                         (l_df.SibSp == s_df.loc[i].SibSp)       &
                         (l_df.Parch == s_df.loc[i].Parch)       &
                         (l_df.Pclass == s_df.loc[i].Pclass)     &
                         (l_df.Embarked == s_df.loc[i].Embarked) &
                         (l_df.Ticket == s_df.loc[i].Ticket)].values[0]

    return df

def result():
    train_data = convert_to_np('train.csv')
    test_data = convert_to_np('test.csv')

    forest = RandomForestClassifier(n_estimators=100)

    forest = forest.fit(train_data[:,1:], train_data[:,0])
    output = forest.predict(test_data)

    final = pd.DataFrame(data=range(892, 1310), columns=['PassengerId'], dtype=int)
    final['Survived'] = output.astype(int)

    final.to_csv('RandomForestOut.csv', index=False)

    test = convert_to_np('titanic_test.csv')

    print forest.score(test[:,1:], test[:,0])

def graph():
    df = pd.read_csv('train.csv')
    # specifies the parameters of our graphs
    fig = plt.figure(figsize=(18,6), dpi=1600) 
    alpha=alpha_scatterplot = 0.2 
    alpha_bar_chart = 0.55
    
    # lets us plot many diffrent shaped graphs together 
    ax1 = plt.subplot2grid((2,3),(0,0))
    # plots a bar graph of those who surived vs those who did not.               
    df.Survived.value_counts().plot(kind='bar', alpha=alpha_bar_chart)
    # this nicely sets the margins in matplotlib to deal with a recent bug 1.3.1
    ax1.set_xlim(-1, 2)
    # puts a title on our graph
    plt.title("Distribution of Survival, (1 = Survived)")    
    
    plt.subplot2grid((2,3),(0,1))
    plt.scatter(df.Survived, df.Age, alpha=alpha_scatterplot)
    # sets the y axis lable
    plt.ylabel("Age")
    # formats the grid line style of our graphs                          
    plt.grid(b=True, which='major', axis='y')  
    plt.title("Survival by Age,  (1 = Survived)")
    
    ax3 = plt.subplot2grid((2,3),(0,2))
    df.Pclass.value_counts().plot(kind="barh", alpha=alpha_bar_chart)
    ax3.set_ylim(-1, len(df.Pclass.value_counts()))
    plt.title("Class Distribution")
    
    plt.subplot2grid((2,3),(1,0), colspan=2)
    # plots a kernel density estimate of the subset of the 1st class passangers's age
    df.Age[df.Pclass == 1].plot(kind='kde')    
    df.Age[df.Pclass == 2].plot(kind='kde')
    df.Age[df.Pclass == 3].plot(kind='kde')
    # plots an axis lable
    plt.xlabel("Age")    
    plt.title("Age Distribution within classes")
    # sets our legend for our graph.
    plt.legend(('1st Class', '2nd Class','3rd Class'),loc='best') 
    
    ax5 = plt.subplot2grid((2,3),(1,2))
    df.Embarked.value_counts().plot(kind='bar', alpha=alpha_bar_chart)
    ax5.set_xlim(-1, len(df.Embarked.value_counts()))
    # specifies the parameters of our graphs
    plt.title("Passengers per boarding location")

    plt.savefig('general_data.png')

def graph_survived():
    df = pd.read_csv('train.csv')
    plt.figure(figsize=(6,4))
    fig, ax = plt.subplots()
    df.Survived.value_counts().plot(kind='barh', color='blue', alpha=.65)
    ax.set_ylim(-1, len(df.Survived.value_counts()))
    plt.title("Survival Breakdown (1=Survived, 0=Died)")
    plt.savefig('survival.png')

def graph_gender():
    df = pd.read_csv('train.csv')
    fig = plt.figure(figsize=(18,6))
    df_male = df.Survived[df.Sex=='male'].value_counts().sort_index()
    df_female = df.Survived[df.Sex == 'female'].value_counts().sort_index()

    ax1 = fig.add_subplot(121)
    df_male.plot(kind='barh', label='Male', alpha=0.55)
    df_female.plot(kind='barh', label='Female', color='#FA2379',alpha=0.55)
    plt.title("Who Survived w/r to Gender. (raw value counts)")
    plt.legend(loc='best')
    ax1.set_ylim(-1, 2)

    ax2 = fig.add_subplot(122)
    (df_male/float(df_male.sum())).plot(kind='barh', label='Male', alpha=0.55)
    (df_female/float(df_female.sum())).plot(kind='barh', color='#FA2379', label='Female', alpha=0.55)
    plt.title("Who survived proportionally w/r Gender")
    plt.legend(loc='best')

    ax2.set_ylim(-1,2)

def graph_scatter():
    SURVIVED_FEMALE_COLOR = '#DB0A5B'
    DEAD_FEMALE_COLOR     = '#663399'
    SURVIVED_MALE_COLOR   = '#446CB3'
    DEAD_MALE_COLOR       = '#2C3E50'

    df = pd.read_csv('train.csv')
    df_survived_male = df[(df.Sex=='male') & (df.Survived==1) ][['Age', 'Pclass']]
    df_dead_male = df[(df.Sex=='male') & (df.Survived==0) ][['Age', 'Pclass']]
    df_survived_female = df[(df.Sex=='female') & (df.Survived==1) ][['Age', 'Pclass']]
    df_dead_female = df[(df.Sex=='female') & (df.Survived==0) ][['Age', 'Pclass']]

    fig = plt.figure(figsize=(40,10), dpi=1600)

    ax1 = fig.add_subplot(151)
    plt.scatter(df_dead_male.Age,
                df_dead_male.Pclass, alpha=0.7,
                color=DEAD_MALE_COLOR, label='Dead Males', s=40)
    plt.title('Dead Males Class/Age')

    ax2 = fig.add_subplot(152)
    plt.scatter(df_survived_male.Age,
                df_survived_male.Pclass, alpha=0.7,
                color=SURVIVED_MALE_COLOR, label='Alive Males', s=40)
    plt.title('Alive Males Class/Age')

    ax3 = fig.add_subplot(153)
    plt.scatter(df_dead_female.Age,
                df_dead_female.Pclass, alpha=0.7,
                color=DEAD_FEMALE_COLOR, label='Dead Females', s=40)
    plt.title('Dead Females Class/Age')

    ax4 = fig.add_subplot(154)
    plt.scatter(df_survived_female.Age,
                df_survived_female.Pclass, alpha=0.7,
                color=SURVIVED_FEMALE_COLOR, label='Alive Females', s=40)
    plt.title('Alive Females Class/Age')

    ax4 = fig.add_subplot(155)
    plt.scatter(df_dead_male.Age,
                df_dead_male.Pclass, alpha=0.7,
                color=DEAD_MALE_COLOR, label='Dead Males', s=40)

    plt.scatter(df_survived_male.Age,
                df_survived_male.Pclass, alpha=0.7,
                color=SURVIVED_MALE_COLOR, label='Alive Males', s=40)

    plt.scatter(df_dead_female.Age,
                df_dead_female.Pclass, alpha=0.7,
                color=DEAD_FEMALE_COLOR, label='Dead Females', s=40)

    plt.scatter(df_survived_female.Age,
                df_survived_female.Pclass, alpha=0.7,
                color=SURVIVED_FEMALE_COLOR, label='Alive Females', s=40)

    axes = plt.gca()
    axes.set_ylim([0.5,3.5])
    axes.set_xlim([0,80])

    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig('mapper.png')
