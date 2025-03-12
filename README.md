## Dataset building

### V2
Now I have a V2 of this project in Python, but I'm not going to share the full code.  
Let me know if you want to take a look!  

- Language: Python.
- Performance: 500 _.txt_ files per second (Macbook Air M2).
- Upgrades:
  - no corrupted data reported so far;
  - very fast data processing with parallelized operations;
  - code best practices implemented;
  - installable package available.

### Description  
This is a data engineering project aimed to build a dataset from online poker games.  I wrote this code in July 2020, when I started to get into the data world. It was mainly developed to study _Spin & Go tournaments_, but it might work more generally. It performs well, but _Regex_ can certainly be improved. Also, today I wouldn’t use so many conditionals and loops that take a lot of time to process.

In summary, this project provides a way to convert hand history files into structured data. Here is an example of a hand history, generated by the poker platforms when you play a hand:
```
MyPokerStudies Hand #216746978471: Tournament #2967580470, $0.23+$0.02 USD Hold'em No Limit - Level I (10/20) - 2020/07/24 14:21:17 BRT [2020/07/24 13:21:17 ET]
Table '2967580470 1' 3-max Seat #1 is the button
Seat 1: villain_A (60 in chips) 
Seat 2: garciamurilo (910 in chips) 
Seat 3: villain_B (530 in chips) 
garciamurilo: posts small blind 10
villain_B: posts big blind 20
*** HOLE CARDS ***
Dealt to garciamurilo [9s Td]
villain_A: folds 
garciamurilo: calls 10
villain_B: checks 
*** FLOP *** [Kd Qh 3c]
garciamurilo: checks 
villain_B: checks 
*** TURN *** [Kd Qh 3c] [8s]
garciamurilo: checks 
villain_B: bets 40
garciamurilo: folds 
Uncalled bet (40) returned to villain_B
villain_B collected 40 from pot
villain_B: doesn't show hand 
*** SUMMARY ***
Total pot 40 | Rake 0 
Board [Kd Qh 3c 8s]
Seat 1: villain_A (button) folded before Flop (didn't bet)
Seat 2: garciamurilo (small blind) folded on the Turn
Seat 3: villain_B (big blind) collected (40)
```
<br>  Processing this example with my _R_ code, you will get a structured data with three rows, one for each player with detailed information of all their actions and profits. Here it is:

buyin|tourn_id|table|hand_id|date|time|table_size|level|playing|seat|name|stack|position|action_pre|action_flop|action_turn|action_river|all_in|cards|board_flop|board_turn|board_river|combination|pot_pre|pot_flop|pot_turn|pot_river|ante|blinds|bet_pre|bet_flop|bet_turn|bet_river|result|balance|
---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
$0.23+$0.02|2967580470|1|216746978471|2020-07-24|14:21:17|3|1|3|1|villain_A|60|BTN|folds|x|x|x|FALSE|--|Kd Qh 3c|8s|0||40|40|80|80|0|0|0|0|0|0|gave up|0|
$0.23+$0.02|2967580470|1|216746978471|2020-07-24|14:21:17|3|1|3|2|garciamurilo|910|SB|calls|checks|checks-folds|x|FALSE|9s Td|Kd Qh 3c|8s|0||40|40|80|80|0|10|20|0|0|0|gave up|-20|
$0.23+$0.02|2967580470|1|216746978471|2020-07-24|14:21:17|3|1|3|3|villain_B|530|BB|checks|checks|bets-doesn't|x|FALSE|--|Kd Qh 3c|8s|0||40|40|80|80|0|20|20|0|40|0|took chips|20|

<br>  

After exploring many possibilities, I found out that it was the perfect structure to get a good exploratory data analysis of my tournaments. So I was able to transform and plot data to get insights about my game and to study my opponents. An example: <br>  
![](https://raw.githubusercontent.com/murilogmamaral/datasetbuilding/main/plot.png)
<br><br>  

### Limitations
Some older poker players have logins with special characters that used to be allowed in the past, and this might affect data quality. I usually find about 2% of corrupted data because of that and I just exclude them. It might be possible to escape special characters at some stage of the process and keep all data safe, but I have been too busy lately to fix this (or any other) point.  
<br>
### Usage 
Download all files and run the _functions.R_ file in your _RStudio_. Then, open the _execute.R_, set the directory of your hand history files and the other variables. After running it, you will get a _result.csv_ with your structured data.  
<br>
### References  
Please go to [kaggle.com/murilogmamaral/online-poker-games](https://www.kaggle.com/murilogmamaral/online-poker-games) if you would like to see the generated dataset from my online poker games and also a brief exploratory data analysis.  
<br>
  
### Data dictionary
column|type|description
---|---|---|
buyin|character|Amount paid (USD) to play the tournament|
tourn_id|integer|Tournament id|
table|integer|Table number reference in the tournament|  
hand_id|integer|Hand id  
date|date|Date of a played hand| 
time|datetime|Time of a played hand|
table_size|integer|Maximum number of players per table|
level|integer|Blinds levels|
playing|integer|Number of players currently in the table|
seat|integer|Seat number of each player|
name|character|Player login|
initial_stack|double|Initial stack of each player|
position|integer|Position of each player|
action_pre|character|Preflop actions of each player| 
action_flop|character|Flop actions of each player| 
action_turn|character|Turn actions of each player| 
action_river|character|River actions of each player| 
all_in|boolean|Informs if a player did an all-in bet (TRUE) or not (FALSE)|
cards|character|Hand of each player (only available if you see it, obviously)| 
board_flop|character|Cards on flop|
board_turn|character|Cards on turn|
board_river|character|Cards on river|
combination|character|Cards combination of each player|
pot_pre|double|Preflop pot size|  
pot_flop|double|Flop pot size|  
pot_turn|double|Turn pot size|  
pot_river|double|River pot size|  
ante|double|Ante paid|
blinds|double|Blinds paid|
bet_pre|double|Bet done on preflop|
bet_flop|double|Bet done on flop|
bet_turn|double|Bet done on turn| 
bet_river|double|Bet done on river| 
result|character|Four categories: 1-won, means a player went to showdown and won; 2-lost, means a player went to showdown and lost; 3-gave up, means a player folded at some point; 4-took chips, means a player took chips without going to showdown.
balance|double|How much a player won or lost after a hand|
