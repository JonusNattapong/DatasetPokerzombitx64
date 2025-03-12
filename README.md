# โครงการวิเคราะห์ข้อมูลโป๊กเกอร์ - Pokerzombitx64

โปรเจกต์นี้นำเสนอชุดเครื่องมือและชุดข้อมูลที่ครอบคลุมสำหรับการวิเคราะห์เกมโป๊กเกอร์ในเชิงลึก ออกแบบมาสำหรับนักวิจัย นักวิทยาศาสตร์ข้อมูล และผู้ที่ชื่นชอบโป๊กเกอร์ที่ต้องการสำรวจกลยุทธ์โป๊กเกอร์ พฤติกรรมผู้เล่น และพลวัตของเกมโดยใช้แนวทางที่ขับเคลื่อนด้วยข้อมูล

## ภาพรวมโครงการ

โปรเจกต์นี้ประกอบด้วยองค์ประกอบหลักหลายส่วน:

1.  **ข้อมูล:** ชุดข้อมูลประวัติมือโป๊กเกอร์ ทั้งแบบดิบและแบบประมวลผลแล้ว
2.  **`pokerdata` Library:** ไลบรารี Python ที่มีโมดูลสำหรับการแยกวิเคราะห์ (parsing) ประเมิน (evaluating) และวิเคราะห์ (analyzing) ข้อมูลโป๊กเกอร์
3.  **ตัวอย่าง:** สคริปต์ตัวอย่างที่สาธิตวิธีการใช้ไลบรารีและชุดข้อมูล
4.  **การทดสอบ:** การทดสอบหน่วย (Unit tests) เพื่อให้มั่นใจในความน่าเชื่อถือและความถูกต้องของโค้ด
5. **เอกสารประกอบ:** ไฟล์ README นี้และเอกสารเพิ่มเติมภายในโค้ด

## ข้อมูล

ชุดข้อมูลถูกจัดระเบียบไว้ในไดเรกทอรีต่อไปนี้:

-   `raw_data/`: ประกอบด้วยไฟล์ประวัติมือโป๊กเกอร์ต้นฉบับที่ยังไม่ได้ประมวลผล ไฟล์เหล่านี้มักจะเป็นไฟล์ข้อความ (`.txt`) ที่ได้จากเว็บไซต์โป๊กเกอร์ออนไลน์หรือซอฟต์แวร์ติดตาม
-   `processed_data/`: ประกอบด้วยข้อมูลที่ทำความสะอาดและจัดรูปแบบแล้ว พร้อมสำหรับการวิเคราะห์ ข้อมูลนี้มักจะอยู่ในรูปแบบ CSV หรือ JSON ทำให้ง่ายต่อการโหลดลงในเครื่องมือวิเคราะห์ข้อมูล เช่น pandas หรือ scikit-learn
-   `metadata/`: ประกอบด้วยข้อมูลเกี่ยวกับชุดข้อมูล เช่น แหล่งที่มาของข้อมูล วิธีการรวบรวม และข้อจำกัดความรับผิดชอบที่เกี่ยวข้อง
- `data/output`: ประกอบด้วยผลลัพธ์การวิเคราะห์ รวมถึงสถิติผู้เล่นและการแสดงผลข้อมูล
- `data/handhistory`: ประกอบด้วยตัวอย่างไฟล์ประวัติมือ

### รูปแบบข้อมูล

ไฟล์ประวัติมือดิบเป็นไฟล์ข้อความที่มีรูปแบบเฉพาะ ซึ่งแตกต่างกันเล็กน้อยขึ้นอยู่กับเว็บไซต์โป๊กเกอร์ โมดูล `parser.py` จะจัดการกับความแตกต่างเหล่านี้และแปลงข้อมูลดิบเป็นรูปแบบมาตรฐาน ข้อมูลที่ประมวลผลแล้วมักจะถูกเก็บไว้ในรูปแบบ CSV หรือ JSON โดยแต่ละแถวแสดงถึงการกระทำเดียวในมือโป๊กเกอร์ (เช่น เดิมพัน, เพิ่มเดิมพัน, เก, หมอบ) ช่องข้อมูลหลักประกอบด้วย:

-   **Hand ID:** ตัวระบุเฉพาะสำหรับแต่ละมือ
-   **Timestamp:** วันที่และเวลาที่เล่น
-   **Game Type:** ประเภทของเกมโป๊กเกอร์ (เช่น "NL Hold'em", "PL Omaha")
-   **Stakes:** จำนวนเงินเดิมพัน (blinds/antes) (เช่น "$1/$2", "0.50/1")
-   **Table Size:** จำนวนผู้เล่นที่โต๊ะ (เช่น 6, 9)
-   **Player Names:** ตัวระบุสำหรับผู้เล่นแต่ละคน (มักจะถูกปิดบังชื่อ)
-   **Player Positions:** ตำแหน่งของผู้เล่นที่โต๊ะ (BTN, SB, BB, UTG, MP, CO)
-   **Player Stack Sizes:** จำนวนชิปที่ผู้เล่นแต่ละคนมีเมื่อเริ่มต้นมือ
-   **Actions:** ลำดับการกระทำของผู้เล่นแต่ละคน (เช่น bet, raise, call, fold, check)
-   **Bet/Raise Amounts:** จำนวนเงินเดิมพันหรือเพิ่มเดิมพัน
-   **Pot Size:** จำนวนชิปทั้งหมดในกองกลางในแต่ละขั้นตอนของมือ
-   **Board Cards:** ไพ่กองกลางที่แจก (flop, turn, river)
-   **Showdown Information:** ผู้เล่นคนใดที่เปิดไพ่และมือที่ชนะ

## `pokerdata` Library

นี่คือส่วนหลักของโปรเจกต์ เป็นไลบรารี Python ที่มีโมดูลสำหรับ:

-   **`parser.py`:** แยกวิเคราะห์ไฟล์ประวัติมือดิบจากเว็บไซต์โป๊กเกอร์ต่างๆ เป็นรูปแบบที่มีโครงสร้าง (เช่น pandas DataFrame หรือรายการพจนานุกรม) จัดการกับรูปแบบไฟล์และความแตกต่างในการบันทึกประวัติมือที่แตกต่างกัน
-   **`card_evaluator.py`:** ประเมินมือโป๊กเกอร์และระบุความแข็งแกร่งของมือ ระบุลำดับชั้นของมือโป๊กเกอร์มาตรฐานทั้งหมดได้อย่างถูกต้อง (Royal Flush, Straight Flush, Four of a Kind, Full House, Flush, Straight, Three of a Kind, Two Pair, Pair, High Card) และจัดการกับกรณีพิเศษ เช่น Wheel Straight (A-2-3-4-5) และ Broadway straight (A-K-Q-J-T) ฟังก์ชัน `compare_hands` ช่วยให้สามารถเปรียบเทียบมือโป๊กเกอร์สองมือได้อย่างแม่นยำ โดยคำนึงถึงไพ่กองกลางด้วย
-   **`hand_range_analyzer.py`:** วิเคราะห์ช่วงมือของผู้เล่นตามการกระทำของพวกเขา โมดูลนี้ช่วยให้เข้าใจแนวโน้มและสไตล์การเล่นของผู้เล่น สามารถคำนวณสถิติ เช่น VPIP (Voluntarily Put Money In Pot) และ PFR (Pre-Flop Raise)
-   **`visualizer.py`:** มีฟังก์ชันสำหรับการสร้างการแสดงภาพข้อมูลโป๊กเกอร์ เช่น กราฟการชนะของผู้เล่นเมื่อเวลาผ่านไป การกระจายช่วงมือ และอื่นๆ
-   **`anonymizer.py`:** ปิดบังชื่อผู้เล่นในข้อมูลประวัติมือเพื่อปกป้องความเป็นส่วนตัว
-   **`db_connector.py`:** (อาจมี) จัดการการเชื่อมต่อกับฐานข้อมูลสำหรับการจัดเก็บและดึงข้อมูลชุดข้อมูลขนาดใหญ่
-   **`ml_analyzer.py`:** วิเคราะห์ข้อมูลโป๊กเกอร์โดยใช้เทคนิคแมชชีนเลิร์นนิ่งเพื่อทำนายผลลัพธ์และระบุรูปแบบที่ซับซ้อน
-   **`__init__.py`:** ทำให้ `pokerdata` เป็นแพ็คเกจ Python

## เทคนิคแมชชีนเลิร์นนิ่งและการวิเคราะห์ขั้นสูง

โปรเจกต์นี้รวมเทคนิคแมชชีนเลิร์นนิ่งและการวิเคราะห์ขั้นสูงเพื่อช่วยในการหาข้อมูลเชิงลึกจากข้อมูลโป๊กเกอร์:

### 1. การจัดกลุ่มผู้เล่น (Player Clustering)

โมดูล `ml_analyzer.py` ใช้อัลกอริธึมการเรียนรู้แบบไม่มีผู้สอน (unsupervised learning) เช่น K-means และ DBSCAN เพื่อจัดกลุ่มผู้เล่นตามสไตล์การเล่นของพวกเขา เช่น:
- ผู้เล่นแบบ Tight-aggressive (TAG)
- ผู้เล่นแบบ Loose-aggressive (LAG)
- ผู้เล่นแบบ Tight-passive (Rock)
- ผู้เล่นแบบ Loose-passive (Calling station)

```python
from pokerdata import ml_analyzer

# จัดกลุ่มผู้เล่นตามสไตล์การเล่น
player_clusters = ml_analyzer.cluster_players(poker_df, n_clusters=4)
ml_analyzer.visualize_player_clusters(player_clusters)
```

### 2. การทำนายการตัดสินใจของผู้เล่น (Decision Prediction)

ใช้โมเดลการเรียนรู้แบบมีผู้สอน (supervised learning) เช่น Random Forest, XGBoost และ Neural Networks เพื่อทำนายการกระทำของผู้เล่นในสถานการณ์ต่างๆ:

```python
from pokerdata import ml_analyzer

# สร้างโมเดลทำนายการตัดสินใจ
model = ml_analyzer.train_decision_model(poker_df, target='action_pre')

# ทำนายว่าผู้เล่นจะ fold, call หรือ raise
prediction = model.predict(new_situation)
```

### 3. การวิเคราะห์ความแข็งแกร่งของช่วงมือ (Range Strength Analysis)

วิเคราะห์ความแข็งแกร่งของช่วงมือในสถานการณ์ต่างๆ โดยใช้การคำนวณแบบ Monte Carlo และการใช้มูลค่าคาดหวัง (expected value):

```python
from pokerdata import ml_analyzer

# วิเคราะห์ความแข็งแกร่งของช่วงมือที่ผู้เล่นเล่นจาก UTG
range_strength = ml_analyzer.analyze_range_strength(
    poker_df,
    player="Player1",
    position="UTG",
    action="raises",
    num_simulations=10000
)
```

### 4. การระบุการเล่นผิดพลาด (Mistake Identification)

ใช้โมเดล GTO (Game Theory Optimal) และการวิเคราะห์มูลค่าคาดหวัง (EV) เพื่อระบุการเล่นที่ผิดพลาด:

```python
from pokerdata import ml_analyzer

# ระบุการเล่นผิดพลาดในประวัติมือโป๊กเกอร์
mistakes = ml_analyzer.identify_mistakes(
    poker_df,
    player="Player1",
    ev_threshold=-0.5  # การตัดสินใจที่ส่งผลให้ EV ลดลงมากกว่า 0.5 big blinds
)
```

### 5. วิวัฒนาการของสไตล์การเล่น (Playing Style Evolution)

ติดตามวิวัฒนาการของสไตล์การเล่นของผู้เล่นโดยใช้การวิเคราะห์อนุกรมเวลา:

```python
from pokerdata import ml_analyzer

# วิเคราะห์วิวัฒนาการของสไตล์การเล่น
style_evolution = ml_analyzer.analyze_style_evolution(
    poker_df,
    player="Player1",
    window_size=500  # จำนวนมือในแต่ละช่วง
)
ml_analyzer.plot_style_evolution(style_evolution)
```

## ตัวอย่าง

ไดเรกทอรี `examples/` มีสคริปต์ตัวอย่างที่สาธิตวิธีการใช้ไลบรารี `pokerdata` และชุดข้อมูล ตัวอย่างเหล่านี้ครอบคลุมงานทั่วไป เช่น:

-   การโหลดและแยกวิเคราะห์ไฟล์ประวัติมือ
-   การประเมินมือโป๊กเกอร์
-   การคำนวณสถิติผู้เล่น (VPIP, PFR, ฯลฯ)
-   การสร้างการแสดงผลข้อมูล
-   การวิเคราะห์ข้อมูลพื้นฐาน

สคริปต์ `examples/basic_analysis.py` เป็นจุดเริ่มต้นที่ดี

## การทดสอบ

ไดเรกทอรี `tests/` มีการทดสอบหน่วยสำหรับไลบรารี `pokerdata` การทดสอบเหล่านี้ทำให้มั่นใจได้ว่าโค้ดทำงานได้อย่างถูกต้อง และช่วยป้องกันการถดถอยเมื่อมีการเปลี่ยนแปลง การทดสอบเขียนขึ้นโดยใช้เฟรมเวิร์ก `unittest`

- `tests/test_card_evaluator.py`: การทดสอบสำหรับโมดูล `card_evaluator.py`
- `tests/test_parser.py`: การทดสอบสำหรับโมดูล `parser.py`

ในการรันการทดสอบทั้งหมด ให้ใช้คำสั่งจากไดเรกทอรีรากของโปรเจกต์:
```bash
python run_all_tests.py
```

## เริ่มต้นใช้งาน

### การติดตั้ง

1.  **โคลนพื้นที่เก็บข้อมูล:**

    ```bash
    git clone https://github.com/yourusername/DatasetPokerzombitx64.git
    cd DatasetPokerzombitx64
    ```
2.  **ติดตั้ง dependencies:**
    ขอแนะนำอย่างยิ่งให้สร้าง virtual environment ก่อนติดตั้ง dependencies:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # บน Linux/macOS
    venv\Scripts\activate  # บน Windows
    ```
    จากนั้นติดตั้งแพ็คเกจที่จำเป็น:
    ```bash
    pip install -r requirements.txt
    ```
    (หมายเหตุ: ไฟล์ `requirements.txt` ยังไม่มีอยู่ในพื้นที่เก็บข้อมูล แต่ควรสร้างขึ้นเพื่อแสดงรายการ dependencies ของโปรเจกต์ เช่น `pandas`, `matplotlib` เป็นต้น)

### การติดตั้ง TensorFlow และคุณสมบัติ Machine Learning

เพื่อใช้คุณสมบัติ machine learning ของชุดข้อมูลนี้ คุณต้องติดตั้ง TensorFlow ก่อน:

### ติดตั้ง TensorFlow

```bash
pip install tensorflow
```

หรือหากคุณต้องการเวอร์ชันที่รองรับ GPU (แนะนำสำหรับการฝึกฝนโมเดลขนาดใหญ่):

```bash
pip install tensorflow-gpu
```

### ติดตั้ง Dependencies อื่นๆ สำหรับ Machine Learning

```bash
pip install scikit-learn matplotlib seaborn
```

### การใช้โมดูล ml_analyzer

หลังจากติดตั้งแล้ว คุณสามารถใช้โมดูล ml_analyzer เพื่อฝึกฝนโมเดลการตัดสินใจและจัดกลุ่มผู้เล่นได้:

```python
from pokerdata import ml_analyzer
import pandas as pd

# โหลดข้อมูล
df = pd.read_csv('processed_data/hands.csv')

# จัดกลุ่มผู้เล่นตามสไตล์การเล่น
clusters = ml_analyzer.cluster_players(df, n_clusters=4)
ml_analyzer.visualize_player_clusters(clusters, save_path='player_clusters.png')

# ฝึกฝนโมเดลทำนายการกระทำ
model_results = ml_analyzer.train_decision_model(df, target='action_pre', player='Player1')
print(f"Model accuracy: {model_results['accuracy']:.2f}")

# วิเคราะห์วิวัฒนาการของสไตล์การเล่น
evolution = ml_analyzer.analyze_style_evolution(df, player='Player1', window_size=500)
ml_analyzer.plot_style_evolution(evolution, save_path='style_evolution.png')
```

### การรันตัวอย่าง
ไปที่ไดเรกทอรี `examples` และรันสคริปต์ `basic_analysis.py`:

```bash
cd examples
python basic_analysis.py
```
สคริปต์นี้จะสาธิตการใช้งานพื้นฐานของไลบรารี

### การรันการทดสอบ
หากต้องการรันการทดสอบทั้งหมด ให้ใช้คำสั่งจากไดเรกทอรีรากของโปรเจกต์:
```bash
python run_all_tests.py
```

## การมีส่วนร่วม

ยินดีต้อนรับการมีส่วนร่วมในการขยายและปรับปรุงชุดข้อมูลนี้! หากคุณมีแนวคิดสำหรับการปรับปรุง คุณสมบัติใหม่ หรือการแก้ไขข้อบกพร่อง โปรดทำตามขั้นตอนเหล่านี้:

1.  **Fork** พื้นที่เก็บข้อมูลบน GitHub
2.  สร้าง **branch** ใหม่สำหรับฟีเจอร์หรือการแก้ไขข้อบกพร่องของคุณ: `git checkout -b your-feature-name`
3.  ทำการเปลี่ยนแปลงและ commit ด้วยข้อความ commit ที่ชัดเจนและสื่อความหมาย
4.  **Push** branch ของคุณไปยัง forked repository: `git push origin your-feature-name`
5.  สร้าง **Pull Request** บนพื้นที่เก็บข้อมูลหลัก อธิบายการเปลี่ยนแปลงของคุณและเหตุผลที่ควรผสานรวม

## ใบอนุญาต

[ระบุใบอนุญาตที่นี่ - เช่น MIT License, Apache License 2.0 เป็นต้น] ดูรายละเอียดในไฟล์ `LICENSE` ขอแนะนำให้ใช้ใบอนุญาตแบบอนุญาต เช่น MIT สำหรับโครงการโอเพนซอร์ส

## ติดต่อ

สำหรับคำถาม ข้อเสนอแนะ หรือปัญหา โปรดติดต่อ [ข้อมูลติดต่อของคุณที่นี่ - เช่น ที่อยู่อีเมลของคุณหรือลิงก์ไปยังตัวติดตามปัญหาของโครงการ]

## กิตติกรรมประกาศ

-   ขอขอบคุณ [ผู้มีส่วนร่วมหรือแหล่งข้อมูล/แรงบันดาลใจ]
-   ขอขอบคุณเป็นพิเศษสำหรับชุมชนโป๊กเกอร์ที่ให้ข้อมูลเชิงลึกและแหล่งข้อมูลอันมีค่า

## พจนานุกรมข้อมูล (โดยละเอียด)

ส่วนนี้จะให้คำอธิบายโดยละเอียดเพิ่มเติมเกี่ยวกับช่องข้อมูลที่พบในไฟล์ข้อมูลที่ประมวลผลแล้ว (CSV/JSON)

| ชื่อฟิลด์          | ชนิดข้อมูล | คำอธิบาย                                                                                                                                                                                                                                                           | ตัวอย่าง             |
| ------------------- | --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- |
| hand_id             | string    | ตัวระบุเฉพาะสำหรับมือ                                                                                                                                                                                                                                      | "HH2900000001"      |
| timestamp           | datetime  | วันที่และเวลาที่เล่น                                                                                                                                                                                                                                   | "2023-10-27 14:30:00"|
| game_type           | string    | ประเภทของเกมโป๊กเกอร์ (เช่น "NL Hold'em", "PL Omaha")                                                                                                                                                                                                                   | "NL Hold'em"        |
| stakes              | string    | จำนวนเงินเดิมพัน (blinds/antes) (เช่น "$1/$2", "0.50/1")                                                                                                                                                                                                                               | "$1/$2"             |
| table_size          | integer   | จำนวนผู้เล่นที่โต๊ะ (เช่น 6, 9)                                                                                                                                                                                                                            | 6                   |
| player_name         | string    | ตัวระบุผู้เล่น (อาจถูกปิดบังชื่อ)                                                                                                                                                                                                                                | "Player1"           |
| position            | string    | ตำแหน่งของผู้เล่นที่โต๊ะ (BTN, SB, BB, UTG, MP, CO)                                                                                                                                                                                                             | "BTN"               |
| starting_stack      | float     | จำนวนชิปของผู้เล่นเมื่อเริ่มต้นมือ                                                                                                                                                                                                                       | 100.00              |
| action              | string    | การกระทำของผู้เล่น (เช่น "bet", "raise", "call", "fold", "check")                                                                                                                                                                                            | "raise"             |
| street              | string    | รอบของมือ ("preflop", "flop", "turn", "river")                                                                                                                                                                                                                 | "preflop"           |
| bet_amount          | float     | จำนวนเงินเดิมพันหรือเพิ่มเดิมพัน                                                                                                                                                                                                                                          | 3.50                |
| pot_size            | float     | ขนาดกองกลางทั้งหมดหลังจากการกระทำ                                                                                                                                                                                                                               | 8.50                |
| board_cards         | string    | ไพ่กองกลางที่แจก (ถ้ามี) รูปแบบ: "card1 card2 card3 card4 card5" (เช่น "As Ks Qs" สำหรับ flop)                                                                                                                                                     | "As Ks Qs"          |
| all_in              | boolean   | ระบุว่าผู้เล่นลงเดิมพันทั้งหมดหรือไม่                                                                                                                                                                                                                   | True                |
| showdown            | boolean   | ระบุว่ามือไปถึง showdown หรือไม่                                                                                                                                                                                                                               | True                |
| winning_hand        | string    | มือที่ชนะใน showdown (ถ้ามี) รูปแบบ: "card1 card2 card3 card4 card5" (เช่น "As Ad Ks Kd 9h")                                                                                                                                                    | "As Ad Ks Kd 9h"    |
| hand_rank           | string    | อันดับของมือที่ชนะ (เช่น "Two Pair", "Straight", "Flush")                                                                                                                                                                                                 | "Straight"          |
| winnings            | float     | จำนวนชิปที่ผู้เล่นชนะในมือ                                                                                                                                                                                                                      | 12.00               |

พจนานุกรมข้อมูลโดยละเอียดนี้จะช่วยให้ผู้ใช้เข้าใจโครงสร้างและเนื้อหาของชุดข้อมูล
