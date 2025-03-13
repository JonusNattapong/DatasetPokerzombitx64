# โครงการวิเคราะห์ข้อมูลโป๊กเกอร์ - Pokerzombitx64

สวัสดีครับ! ยินดีต้อนรับสู่โปรเจกต์ Pokerzombitx64 โปรเจกต์นี้เป็นชุดเครื่องมือและข้อมูลที่ครอบคลุมสำหรับการวิเคราะห์เกมโป๊กเกอร์ในเชิงลึก ถูกออกแบบมาเพื่อเป็นแหล่งข้อมูลสำหรับนักวิจัย, นักวิทยาศาสตร์ข้อมูล, ผู้ที่สนใจใน Machine Learning, และผู้ที่ชื่นชอบในเกมโป๊กเกอร์ ที่ต้องการจะศึกษา, วิจัย, และพัฒนาต่อยอดในด้านต่างๆ ของเกมโป๊กเกอร์ โดยใช้ข้อมูลเป็นตัวขับเคลื่อน

**เป้าหมายหลักของโปรเจกต์นี้คือการเปิดโอกาสให้ทุกคนได้:**

*   **ศึกษา:** ทำความเข้าใจกลยุทธ์โป๊กเกอร์, พฤติกรรมผู้เล่น, และพลวัตของเกมอย่างละเอียด
*   **วิจัย:** ค้นคว้าและทดลองแนวคิดใหม่ๆ ในการวิเคราะห์ข้อมูลโป๊กเกอร์
*   **พัฒนา:** สร้างโมเดล AI, เครื่องมือวิเคราะห์, หรือแอปพลิเคชันใหม่ๆ ที่ต่อยอดจากโปรเจกต์นี้

## ภาพรวมโครงการ

โปรเจกต์นี้ประกอบด้วยองค์ประกอบหลักหลายส่วน:

1.  **ข้อมูล (Data):** ชุดข้อมูลประวัติมือโป๊กเกอร์ (Hand History) ทั้งแบบดิบ (Raw) และแบบประมวลผลแล้ว (Processed)
2.  **`pokerdata` Library:** ไลบรารี Python ที่มีโมดูลสำหรับการแยกวิเคราะห์ (Parsing), ประเมิน (Evaluating), และวิเคราะห์ (Analyzing) ข้อมูลโป๊กเกอร์
3.  **ตัวอย่าง (Examples):** สคริปต์ตัวอย่างที่สาธิตวิธีการใช้ไลบรารีและชุดข้อมูล
4.  **การทดสอบ (Tests):** การทดสอบหน่วย (Unit tests) เพื่อให้มั่นใจในความน่าเชื่อถือและความถูกต้องของโค้ด
5.  **เอกสารประกอบ (Documentation):** ไฟล์ README นี้ (ฉบับภาษาไทย) และเอกสารเพิ่มเติมภายในโค้ด (Docstrings)

## ข้อมูล (Data)

ชุดข้อมูลถูกจัดระเบียบไว้ในไดเรกทอรีต่อไปนี้:

*   `raw_data/`: (อาจไม่มีใน Repository เริ่มต้น) ประกอบด้วยไฟล์ประวัติมือโป๊กเกอร์ต้นฉบับที่ยังไม่ได้ประมวลผล ไฟล์เหล่านี้มักจะเป็นไฟล์ข้อความ (`.txt`) ที่ได้จากเว็บไซต์โป๊กเกอร์ออนไลน์หรือซอฟต์แวร์ติดตาม
*   `processed_data/`: (อาจไม่มีใน Repository เริ่มต้น) ประกอบด้วยข้อมูลที่ทำความสะอาดและจัดรูปแบบแล้ว พร้อมสำหรับการวิเคราะห์ ข้อมูลนี้มักจะอยู่ในรูปแบบ CSV หรือ JSON ทำให้ง่ายต่อการโหลดลงในเครื่องมือวิเคราะห์ข้อมูล เช่น pandas หรือ scikit-learn
*   `metadata/`: (อาจไม่มีใน Repository เริ่มต้น) ประกอบด้วยข้อมูลเกี่ยวกับชุดข้อมูล เช่น แหล่งที่มาของข้อมูล วิธีการรวบรวม และข้อจำกัดความรับผิดชอบที่เกี่ยวข้อง
*   `data/output`: ประกอบด้วยผลลัพธ์การวิเคราะห์ รวมถึงสถิติผู้เล่นและการแสดงผลข้อมูล
*   `data/handhistory`: ประกอบด้วยตัวอย่างไฟล์ประวัติมือ

### รูปแบบข้อมูล

ไฟล์ประวัติมือดิบเป็นไฟล์ข้อความที่มีรูปแบบเฉพาะ ซึ่งแตกต่างกันเล็กน้อยขึ้นอยู่กับเว็บไซต์โป๊กเกอร์ โมดูล `parser.py` ใน `pokerdata` Library จะจัดการกับความแตกต่างเหล่านี้และแปลงข้อมูลดิบเป็นรูปแบบมาตรฐาน ข้อมูลที่ประมวลผลแล้วมักจะถูกเก็บไว้ในรูปแบบ CSV หรือ JSON โดยแต่ละแถวแสดงถึงการกระทำเดียวในมือโป๊กเกอร์ (เช่น เดิมพัน, เพิ่มเดิมพัน, เก, หมอบ) ช่องข้อมูลหลักประกอบด้วย:

| ชื่อฟิลด์          | ชนิดข้อมูล | คำอธิบาย                                                                                                | ตัวอย่าง             |
| ------------------- | --------- | -------------------------------------------------------------------------------------------------------- | ------------------- |
| hand_id             | string    | ตัวระบุเฉพาะสำหรับแต่ละมือ                                                                                | "HH2900000001"      |
| timestamp           | datetime  | วันที่และเวลาที่เล่น                                                                                      | "2023-10-27 14:30:00"|
| game_type           | string    | ประเภทของเกมโป๊กเกอร์ (เช่น "NL Hold'em", "PL Omaha")                                                       | "NL Hold'em"        |
| stakes              | string    | จำนวนเงินเดิมพัน (blinds/antes) (เช่น "$1/$2", "0.50/1")                                                   | "$1/$2"             |
| table_size          | integer   | จำนวนผู้เล่นที่โต๊ะ (เช่น 6, 9)                                                                          | 6                   |
| player_name         | string    | ตัวระบุผู้เล่น (อาจถูกปิดบังชื่อ)                                                                        | "Player1"           |
| position            | string    | ตำแหน่งของผู้เล่นที่โต๊ะ (BTN, SB, BB, UTG, MP, CO)                                                         | "BTN"               |
| starting_stack      | float     | จำนวนชิปของผู้เล่นเมื่อเริ่มต้นมือ                                                                        | 100.00              |
| action              | string    | การกระทำของผู้เล่น (เช่น "bet", "raise", "call", "fold", "check")                                          | "raise"             |
| street              | string    | รอบของมือ ("preflop", "flop", "turn", "river")                                                             | "preflop"           |
| bet_amount          | float     | จำนวนเงินเดิมพันหรือเพิ่มเดิมพัน                                                                            | 3.50                |
| pot_size            | float     | ขนาดกองกลางทั้งหมดหลังจากการกระทำ                                                                         | 8.50                |
| board_cards         | string    | ไพ่กองกลางที่แจก (ถ้ามี) รูปแบบ: "card1 card2 card3 card4 card5" (เช่น "As Ks Qs" สำหรับ flop)              | "As Ks Qs"          |
| all_in              | boolean   | ระบุว่าผู้เล่นลงเดิมพันทั้งหมดหรือไม่                                                                      | True                |
| showdown            | boolean   | ระบุว่ามือไปถึง showdown หรือไม่                                                                           | True                |
| winning_hand        | string    | มือที่ชนะใน showdown (ถ้ามี) รูปแบบ: "card1 card2 card3 card4 card5" (เช่น "As Ad Ks Kd 9h")                | "As Ad Ks Kd 9h"    |
| hand_rank           | string    | อันดับของมือที่ชนะ (เช่น "Two Pair", "Straight", "Flush")                                                 | "Straight"          |
| winnings            | float     | จำนวนชิปที่ผู้เล่นชนะในมือ                                                                                | 12.00               |

## `pokerdata` Library

`pokerdata` เป็นหัวใจหลักของโปรเจกต์นี้ เป็นไลบรารี Python ที่สร้างขึ้นมาโดยเฉพาะสำหรับการวิเคราะห์ข้อมูลโป๊กเกอร์ ประกอบด้วยโมดูลต่างๆ ที่ทำงานร่วมกันได้อย่างมีประสิทธิภาพ:

*   **`parser.py`:** โมดูลนี้ทำหน้าที่ "อ่าน" และ "แปล" ไฟล์ประวัติมือดิบจากเว็บไซต์โป๊กเกอร์ต่างๆ ให้เป็นรูปแบบข้อมูลที่เป็นมาตรฐาน (เช่น pandas DataFrame หรือ Python Dictionary) ทำให้ง่ายต่อการนำไปวิเคราะห์ต่อ
*   **`card_evaluator.py`:** โมดูลนี้คือ "ผู้เชี่ยวชาญด้านไพ่" สามารถประเมินความแข็งแกร่งของมือโป๊กเกอร์ได้อย่างแม่นยำ ไม่ว่าจะเป็น Royal Flush, Straight Flush, หรือ High Card ก็สามารถจัดอันดับได้อย่างถูกต้อง รวมถึงกรณีพิเศษต่างๆ
    *   มีฟังก์ชัน `compare_hands` ที่เปรียบเทียบความแข็งแกร่งของมือสองมือได้ โดยคำนึงถึงไพ่กองกลาง (Community Cards) ด้วย
*   **`hand_range_analyzer.py`:** โมดูลนี้คือ "นักสืบพฤติกรรม" จะวิเคราะห์ช่วงมือ (Hand Range) ของผู้เล่นแต่ละคน จากรูปแบบการเล่นที่ผ่านมา ทำให้เราเห็นแนวโน้มและสไตล์การเล่นของผู้เล่นแต่ละคน
    *   คำนวณสถิติสำคัญๆ เช่น VPIP (Voluntarily Put Money In Pot - ความถี่ที่ผู้เล่นลงเงินในพอตโดยสมัครใจ) และ PFR (Pre-Flop Raise - ความถี่ที่ผู้เล่นเก (Raise) ก่อนเปิดไพ่กองกลาง)
*   **`visualizer.py`:** โมดูลนี้คือ "นักสร้างภาพ" จะสร้างแผนภาพและกราฟต่างๆ ที่ช่วยให้เห็นภาพรวมของข้อมูลโป๊กเกอร์ได้ง่ายขึ้น เช่น กราฟแสดงผลกำไร/ขาดทุนของผู้เล่น, การกระจายตัวของช่วงมือ, และอื่นๆ
*   **`anonymizer.py`:** โมดูลนี้คือ "ผู้พิทักษ์ความเป็นส่วนตัว" จะลบหรือปิดบังชื่อผู้เล่นในข้อมูลประวัติมือ เพื่อปกป้องข้อมูลส่วนบุคคล
*   **`db_connector.py`:** (อาจมีหรือไม่มีก็ได้) โมดูลนี้ทำหน้าที่เชื่อมต่อกับฐานข้อมูล (Database) ใช้สำหรับจัดเก็บและดึงข้อมูลโป๊กเกอร์จำนวนมากๆ
*   **`ml_analyzer.py`:** "มันสมองกล" ของโปรเจกต์ ใช้เทคนิค Machine Learning ในการวิเคราะห์ข้อมูลโป๊กเกอร์ เพื่อทำนายผลลัพธ์และค้นหารูปแบบที่ซับซ้อน
*   **`__init__.py`:** ไฟล์นี้ทำให้ `pokerdata` เป็นแพ็คเกจ Python ที่สมบูรณ์

## โมเดล ZomPokerX64

ZomPokerX64 คือ "เพชรยอดมงกุฎ" ของโปรเจกต์ เป็นโมเดล AI โป๊กเกอร์ขั้นสูงที่พัฒนาโดย Zombitx64 ผสมผสานกลยุทธ์หลากหลายรูปแบบ:

*   **Rule-Based Strategy:** ใช้กฎเกณฑ์ที่กำหนดไว้ล่วงหน้า (เช่น ความแข็งแกร่งของมือ, ตำแหน่ง)
*   **Neural Networks:** ใช้โครงข่ายประสาทเทียม (Neural Networks) ในการตัดสินใจ
*   **Opponent Modeling:** สร้างแบบจำลองของคู่ต่อสู้ เพื่อปรับกลยุทธ์ให้เหมาะสม
*   **Monte Carlo Simulations:** ใช้การจำลอง Monte Carlo เพื่อคำนวณความน่าจะเป็นและ Equity

**คุณสมบัติเด่น:**

*   มีการประเมินความแข็งแกร่งของมือ (Hand Strength) ที่คำนวณไว้ล่วงหน้า
*   ใช้ Neural Networks ในการตัดสินใจ ทำให้สามารถเรียนรู้และปรับตัวได้
*   มีระบบจำลองคู่ต่อสู้ (Opponent Modeling) ทำให้ AI สามารถปรับกลยุทธ์ตามสไตล์การเล่นของคู่ต่อสู้ได้
*   ใช้การจำลอง Monte Carlo เพื่อคำนวณ Equity (โอกาสในการชนะ) ได้อย่างแม่นยำ
*   มีรูปแบบการเดิมพัน (Betting Patterns) ที่ปรับเปลี่ยนได้

## เทคนิค Machine Learning และการวิเคราะห์ขั้นสูง

โปรเจกต์นี้ไม่ได้มีแค่เครื่องมือพื้นฐาน แต่ยังรวมเทคนิค Machine Learning และการวิเคราะห์ขั้นสูง เพื่อให้ได้ข้อมูลเชิงลึก (Insights) จากข้อมูลโป๊กเกอร์:

### 1. การจัดกลุ่มผู้เล่น (Player Clustering)

ใช้ `ml_analyzer.py` เพื่อจัดกลุ่มผู้เล่นตามสไตล์การเล่น โดยใช้อัลกอริทึม Unsupervised Learning เช่น K-means และ DBSCAN ทำให้เราสามารถแบ่งผู้เล่นออกเป็นกลุ่มๆ เช่น:

*   **Tight-Aggressive (TAG):** ผู้เล่นที่เล่นเฉพาะมือดีๆ และเล่นอย่างดุดัน
*   **Loose-Aggressive (LAG):** ผู้เล่นที่เล่นหลายมือ และเล่นอย่างดุดัน
*   **Tight-Passive (Rock):** ผู้เล่นที่เล่นเฉพาะมือดีๆ และเล่นอย่างระมัดระวัง
*   **Loose-Passive (Calling Station):** ผู้เล่นที่เล่นหลายมือ และมักจะ Call มากกว่า Raise

```python
from pokerdata import ml_analyzer

# จัดกลุ่มผู้เล่น
player_clusters = ml_analyzer.cluster_players(poker_df, n_clusters=4)
ml_analyzer.visualize_player_clusters(player_clusters) # แสดงผลการจัดกลุ่ม
```

### 2. การทำนายการตัดสินใจของผู้เล่น (Decision Prediction)

ใช้ Supervised Learning Models เช่น Random Forest, XGBoost, และ Neural Networks เพื่อทำนายว่าผู้เล่นจะตัดสินใจอย่างไร (Fold, Call, Raise, Bet) ในสถานการณ์ต่างๆ

```python
from pokerdata import ml_analyzer

# สร้างโมเดลทำนาย
model = ml_analyzer.train_decision_model(poker_df, target='action_pre')

# ทำนายการตัดสินใจในสถานการณ์ใหม่
prediction = model.predict(new_situation)
```

### 3. การวิเคราะห์ความแข็งแกร่งของช่วงมือ (Range Strength Analysis)

วิเคราะห์ว่าช่วงมือ (Hand Range) ของผู้เล่นแข็งแกร่งแค่ไหนในแต่ละสถานการณ์ โดยใช้การคำนวณแบบ Monte Carlo และ Expected Value (EV)

```python
from pokerdata import ml_analyzer

# วิเคราะห์ความแข็งแกร่งของช่วงมือ
range_strength = ml_analyzer.analyze_range_strength(
    poker_df,
    player="Player1",
    position="UTG",
    action="raises",
    num_simulations=10000
)
```

### 4. การระบุการเล่นผิดพลาด (Mistake Identification)

ใช้ Game Theory Optimal (GTO) Models และการวิเคราะห์ Expected Value (EV) เพื่อระบุว่าการตัดสินใจใดของผู้เล่นที่เป็น "การเล่นที่ผิดพลาด" (Mistake)

```python
from pokerdata import ml_analyzer

# ระบุการเล่นผิดพลาด
mistakes = ml_analyzer.identify_mistakes(
    poker_df,
    player="Player1",
    ev_threshold=-0.5  # กำหนดเกณฑ์ EV
)
```

### 5. วิวัฒนาการของสไตล์การเล่น (Playing Style Evolution)

ติดตามการเปลี่ยนแปลงสไตล์การเล่นของผู้เล่นเมื่อเวลาผ่านไป โดยใช้ Time Series Analysis

```python
from pokerdata import ml_analyzer

# วิเคราะห์วิวัฒนาการ
style_evolution = ml_analyzer.analyze_style_evolution(
    poker_df,
    player="Player1",
    window_size=500  # กำหนดขนาด Window
)
ml_analyzer.plot_style_evolution(style_evolution) # แสดงผล
```

## ตัวอย่างการใช้งาน (Examples)

ในไดเรกทอรี `examples/` มีสคริปต์ตัวอย่างที่สาธิตวิธีการใช้งาน `pokerdata` Library และชุดข้อมูล ตัวอย่างเหล่านี้ครอบคลุมการใช้งานทั่วไป เช่น:

*   การโหลดและแยกวิเคราะห์ไฟล์ประวัติมือ
*   การประเมินความแข็งแกร่งของมือโป๊กเกอร์
*   การคำนวณสถิติผู้เล่น (VPIP, PFR, ฯลฯ)
*   การสร้างแผนภาพและกราฟ
*   การวิเคราะห์ข้อมูลเบื้องต้น

สคริปต์ `examples/basic_analysis.py` เป็นจุดเริ่มต้นที่ดีในการศึกษา

## การทดสอบ (Tests)

ในไดเรกทอรี `tests/` มี Unit Tests สำหรับ `pokerdata` Library การทดสอบเหล่านี้จะช่วยให้มั่นใจว่าโค้ดทำงานได้อย่างถูกต้อง และป้องกันไม่ให้เกิดข้อผิดพลาดเมื่อมีการแก้ไขโค้ดในอนาคต การทดสอบเขียนขึ้นโดยใช้เฟรมเวิร์ก `unittest`

*   `tests/test_card_evaluator.py`: ทดสอบโมดูล `card_evaluator.py`
*   `tests/test_parser.py`: ทดสอบโมดูล `parser.py`

**วิธีรันการทดสอบ:**

จากไดเรกทอรีหลักของโปรเจกต์ รันคำสั่ง:

```bash
python run_all_tests.py
```

## เริ่มต้นใช้งาน

### การติดตั้ง

1.  **โคลน (Clone) Repository:**

    ```bash
    git clone https://github.com/yourusername/DatasetPokerzombitx64.git
    cd DatasetPokerzombitx64
    ```
2.  **ติดตั้ง Dependencies:**

    แนะนำให้สร้าง Virtual Environment ก่อน:

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # Linux/macOS
    venv\Scripts\activate  # Windows
    ```

    จากนั้นติดตั้งแพ็คเกจที่จำเป็น:

    ```bash
    pip install -r requirements.txt
    ```

    (หมายเหตุ: ไฟล์ `requirements.txt` จะต้องถูกสร้างขึ้น และระบุรายการ Dependencies ทั้งหมดของโปรเจกต์ เช่น `pandas`, `matplotlib`, `scikit-learn`, `tensorflow`, `safetensors` ฯลฯ)

### การติดตั้ง TensorFlow และคุณสมบัติ Machine Learning

หากต้องการใช้คุณสมบัติ Machine Learning ของโปรเจกต์นี้ คุณจะต้องติดตั้ง TensorFlow:

```bash
pip install tensorflow
```

หรือถ้าต้องการรุ่นที่รองรับ GPU (แนะนำสำหรับการเทรนโมเดลขนาดใหญ่):

```bash
pip install tensorflow-gpu
```

และติดตั้ง Dependencies อื่นๆ ที่จำเป็นสำหรับ Machine Learning:

```bash
pip install scikit-learn matplotlib seaborn
```

### การใช้งานโมดูล `ml_analyzer`

หลังจากติดตั้ง Dependencies ทั้งหมดแล้ว คุณสามารถใช้งานโมดูล `ml_analyzer` ได้:

```python
from pokerdata import ml_analyzer
import pandas as pd

# โหลดข้อมูล
df = pd.read_csv('processed_data/hands.csv') # แก้ไข path ให้ถูกต้อง

# จัดกลุ่มผู้เล่น
clusters = ml_analyzer.cluster_players(df, n_clusters=4)
ml_analyzer.visualize_player_clusters(clusters, save_path='player_clusters.png')

# ฝึกฝนโมเดลทำนาย
model_results = ml_analyzer.train_decision_model(df, target='action_pre', player='Player1')
print(f"Model accuracy: {model_results['accuracy']:.2f}")

# วิเคราะห์วิวัฒนาการ
evolution = ml_analyzer.analyze_style_evolution(df, player='Player1', window_size=500)
ml_analyzer.plot_style_evolution(evolution, save_path='style_evolution.png')
```

### การรันตัวอย่าง

ไปที่ไดเรกทอรี `examples` และรันสคริปต์ `basic_analysis.py`:

```bash
cd examples
python basic_analysis.py
```

### การรันการทดสอบ

จากไดเรกทอรีหลักของโปรเจกต์ รัน:

```bash
python run_all_tests.py
```

## การมีส่วนร่วม (Contributions)

เรายินดีต้อนรับทุกท่านที่มีส่วนร่วมในการพัฒนาโปรเจกต์นี้ให้ดียิ่งขึ้น! ไม่ว่าจะเป็นการเสนอไอเดียใหม่ๆ, เพิ่มฟีเจอร์, แก้ไขข้อบกพร่อง, หรือปรับปรุงเอกสาร

**ขั้นตอนการมีส่วนร่วม:**

1.  **Fork** Repository นี้บน GitHub
2.  สร้าง **Branch** ใหม่: `git checkout -b your-feature-name`
3.  ทำการเปลี่ยนแปลงและ **Commit**: `git commit -m "คำอธิบายการเปลี่ยนแปลง"`
4.  **Push** Branch ของคุณ: `git push origin your-feature-name`
5.  สร้าง **Pull Request**

## ใบอนุญาต (License)

โปรเจกต์นี้อยู่ภายใต้ [ระบุชื่อ License - เช่น MIT License] ดูรายละเอียดในไฟล์ `LICENSE`

## ติดต่อ (Contact)

หากมีคำถาม ข้อเสนอแนะ หรือพบปัญหา โปรดติดต่อ [ข้อมูลติดต่อ - เช่น อีเมล]

## กิตติกรรมประกาศ (Acknowledgements)

*   ขอขอบคุณ [ผู้มีส่วนร่วม หรือแหล่งข้อมูล/แรงบันดาลใจ]
*   ขอขอบคุณเป็นพิเศษสำหรับชุมชนโป๊กเกอร์

**มาร่วมกันทำให้โปรเจกต์นี้เป็นแหล่งข้อมูลอันล้ำค่าสำหรับวงการโป๊กเกอร์และการวิเคราะห์ข้อมูล!**
