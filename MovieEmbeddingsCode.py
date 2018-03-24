{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Question 1: Movie Embeddings"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# i) Computing users who liked both movies i and j "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "ratings = pd.read_csv(\"movieratings.csv\", names=[\"MovieID\", \"UserID\", \"Rating\"])\n",
    "ratings = ratings[ratings['Rating']==1]\n",
    "#print(ratings)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "942\n"
     ]
    }
   ],
   "source": [
    "def xij (ratings):\n",
    "    DistinctUsers = ratings.UserID.unique()\n",
    "    size = (1682,1682)\n",
    "    X = np.zeros(size, dtype='int')\n",
    "    for userID in DistinctUsers:\n",
    "        likedMovies = ratings.loc[ratings[\"UserID\"] == userID].MovieID\n",
    "        for i in likedMovies:\n",
    "            for j in likedMovies:\n",
    "                X[i][j]=X[i][j]+1;\n",
    "    return X\n",
    "print(len(ratings.UserID.unique()))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# ii) Optimize using gradient descent"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def train(X): # Each coordinate of each vector is i.i.d random uniform between [−0.7, 0.7]\n",
    "    V=np.matrix(np.random.uniform(-0.7,0.7, size=(1682, 300)))\n",
    "    alpha=0.00001\n",
    "    cost=[]\n",
    "    iteration=1000\n",
    "    for i in range(iteration):\n",
    "        VT=V.T\n",
    "        error= np.dot(V,VT) - np.array(X)\n",
    "        np.fill_diagonal(error,0)\n",
    "        cost1=np.sum(np.square(error))\n",
    "        if i%100==0:\n",
    "            print(\"iterartion %d: loss %f\" %(i,cost1))\n",
    "        dV = np.dot(VT,error)\n",
    "        cost.append(cost1)\n",
    "        dV+= VT\n",
    "        f= dV*alpha\n",
    "        V-=f.T\n",
    "    return V,cost"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def train1(X): #all the vectors are zeros\n",
    "    V=np.zeros((1682, 300))\n",
    "    alpha=0.00001\n",
    "    cost=[]\n",
    "    iteration=1000\n",
    "    for i in range(iteration):\n",
    "        VT=V.T\n",
    "        error= np.dot(V,VT) - np.array(X)\n",
    "        np.fill_diagonal(error,0)\n",
    "        cost1=np.sum(np.square(error))\n",
    "        if i%100==0:\n",
    "            print(\"iterartion %d: loss %f\" %(i,cost1))\n",
    "        dV = np.dot(VT,error)\n",
    "        cost.append(cost1)\n",
    "        dV+= VT\n",
    "        f= dV*alpha\n",
    "        V-=f.T\n",
    "    return V,cost"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "iterartion 0: loss 191091974.868423\n",
      "iterartion 100: loss 15476105.640488\n",
      "iterartion 200: loss 8091471.573201\n",
      "iterartion 300: loss 5473987.723374\n",
      "iterartion 400: loss 4094360.822695\n",
      "iterartion 500: loss 3262784.001697\n",
      "iterartion 600: loss 2707909.380456\n",
      "iterartion 700: loss 2305806.747947\n",
      "iterartion 800: loss 1998207.139839\n",
      "iterartion 900: loss 1754534.741777\n",
      "iterartion 0: loss 168297394.000000\n",
      "iterartion 100: loss 168297394.000000\n",
      "iterartion 200: loss 168297394.000000\n",
      "iterartion 300: loss 168297394.000000\n",
      "iterartion 400: loss 168297394.000000\n",
      "iterartion 500: loss 168297394.000000\n",
      "iterartion 600: loss 168297394.000000\n",
      "iterartion 700: loss 168297394.000000\n",
      "iterartion 800: loss 168297394.000000\n",
      "iterartion 900: loss 168297394.000000\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n7AhXVthxLgAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x1048b2f60>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline\n",
    "def plotCost(costs):\n",
    "    plt.plot(costs)\n",
    "    \n",
    "X = xij(ratings)\n",
    "model, costList = train(X)\n",
    "plotCost(costList)\n",
    "model1, costList1 = train1(X)\n",
    "plotCost(costList1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Top 10 movies given movie ‘Aladdin’"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "movies = pd.read_csv(\"movies.csv\", names=[\"Movies\"], encoding = \"ISO-8859-1\")\n",
    "movies[\"Movies\"] = movies[\"Movies\"].map(lambda x : x.split(\"|\")[1].strip())# separate columns in csv\n",
    "alladinID = movies[movies[\"Movies\"]=='Aladdin (1992)'].index[0]\n",
    "toyStoryID = movies[movies[\"Movies\"]=='Toy Story (1995)'].index[0]\n",
    "homeAloneID = movies[movies[\"Movies\"]=='Home Alone (1990)'].index[0]\n",
    "\n",
    "from scipy import spatial\n",
    "tree = spatial.KDTree(model)\n",
    "top10Alladin = tree.query(model[alladinID], k=10)[1][0]\n",
    "top10toyStory = tree.query(model[toyStoryID], k=1682)[1][0]\n",
    "top10homeAlone = tree.query(model[homeAloneID], k=1682)[1][0]\n",
    "TS_HA = tree.query(model[top10toyStory == top10homeAlone],k=10)[1][0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Aladdin (1992)\n",
      "Crow\n",
      "Shawshank Redemption\n",
      "Faster Pussycat! Kill! Kill! (1965)\n",
      "White Man's Burden (1995)\n",
      "Strange Days (1995)\n",
      "Maverick (1994)\n",
      "Father of the Bride Part II (1995)\n",
      "Mrs. Doubtfire (1993)\n",
      "Young Guns (1988)\n"
     ]
    }
   ],
   "source": [
    "for item in top10Alladin:\n",
    "    print(movies.iloc[item].Movies)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "American Dream (1990)\n",
      "Murder in the First (1995)\n",
      "Albino Alligator (1996)\n",
      "I\n",
      "Butcher Boy\n",
      "Loaded (1994)\n",
      "My Family (1995)\n",
      "My Crazy Life (Mi vida loca) (1993)\n",
      "North (1994)\n",
      "Man from Down Under\n"
     ]
    }
   ],
   "source": [
    "for item in TS_HA:\n",
    "    print(movies.iloc[item].Movies)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
