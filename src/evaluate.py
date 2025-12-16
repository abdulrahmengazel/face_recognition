
import psycopg2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import random
from config import DB_CONFIG


# دالة لحساب المسافة الإقليدية (مطابقة لـ pgvector <->)
def euclidean_distance(v1, v2):
    return np.linalg.norm(v1 - v2)


# دالة لحساب تشابه جيب التمام (مطابقة لـ pgvector <=>)
# FaceNet يفضل هذه المسافة عادة
def cosine_distance(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return 1 - (dot_product / (norm_v1 * norm_v2))


def evaluate_thresholds():
    print("🔄 Connecting to Database and fetching embeddings...")

    # ... داخل دالة evaluate_thresholds ...

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # --- التصحيح: استخدام JOIN لجلب الاسم من جدول people والبصمة من face_encodings ---
    query = """
            SELECT p.name, f.encoding_facenet
            FROM face_encodings f
                     JOIN people p ON f.person_id = p.id
            WHERE f.encoding_facenet IS NOT NULL; \
            """

    cur.execute(query)
    rows = cur.fetchall()
    conn.close()

    # ... باقي الكود كما هو ...

    if not rows:
        print("❌ No data found in database!")
        return

    print(f"✅ Loaded {len(rows)} face embeddings. Processing pairs...")

    # تنظيم البيانات: { "George_Bush": [vec1, vec2, ...], ... }
    people_data = {}
    for name, encoding in rows:
        # --- بداية التعديل: معالجة النص القادم من قاعدة البيانات ---
        if isinstance(encoding, str):
            # 1. إزالة الأقواس المربعة [ ]
            cleaned_str = encoding.replace('[', '').replace(']', '')
            # 2. تحويل النص إلى قائمة أرقام (Floats)
            encoding = [float(x) for x in cleaned_str.split(',') if x.strip()]
        # --- نهاية التعديل ---

        # الآن التحويل لـ Numpy سيتم بنجاح
        vec = np.array(encoding, dtype=np.float32)

        if name not in people_data:
            people_data[name] = []
        people_data[name].append(vec)

    positive_distances = []  # نفس الشخص
    negative_distances = []  # أشخاص مختلفين

    names_list = list(people_data.keys())

    # 1. حساب المسافات لنفس الشخص (Positives)
    for name, vecs in people_data.items():
        if len(vecs) > 1:
            # إنشاء كل الاحتمالات الممكنة بين صور نفس الشخص
            for v1, v2 in combinations(vecs, 2):
                dist = euclidean_distance(v1, v2)  # غيّر إلى cosine_distance لو أردت
                positive_distances.append(dist)

    # 2. حساب مسافات لأشخاص مختلفين (Negatives)
    # نأخذ عينة عشوائية لتوفير الوقت (مثلاً 10,000 زوج)
    num_negatives = min(len(positive_distances) * 2, 20000)
    if num_negatives == 0: num_negatives = 1000

    print(f"📊 Calculating {len(positive_distances)} positive pairs and ~{num_negatives} negative pairs...")

    for _ in range(num_negatives):
        name1, name2 = random.sample(names_list, 2)
        # تأكد أن الاسمين مختلفين
        while name1 == name2:
            name1, name2 = random.sample(names_list, 2)

        vec1 = random.choice(people_data[name1])
        vec2 = random.choice(people_data[name2])

        dist = euclidean_distance(vec1, vec2)
        negative_distances.append(dist)

    # --- الرسم البياني ---
    plt.figure(figsize=(12, 6))

    # رسم توزيع المسافات لنفس الشخص (أخضر)
    sns.kdeplot(positive_distances, fill=True, color='green', label='Same Person (Match)')

    # رسم توزيع المسافات لأشخاص مختلفين (أحمر)
    sns.kdeplot(negative_distances, fill=True, color='red', label='Different People (No Match)')

    plt.title('FaceNet Distance Distribution (LFW Data)')
    plt.xlabel('Distance (Euclidean)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # حفظ الرسم
    plt.savefig("threshold_analysis.png")
    print("📈 Graph saved as 'threshold_analysis.png'. Check it to pick your threshold!")
    plt.show()

    # --- اقتراح أفضل Threshold ---
    # هو الرقم الذي يفصل بين التوزيعين بأقل خطأ
    suggested_threshold = 0
    min_overlap = float('inf')

    # فحص نطاق مسافات تجريبي
    for t in np.arange(0, 2.0, 0.01):
        # False Negatives: أشخاص نفس بعض، لكن مسافتهم أكبر من العتبة
        fn = sum(1 for d in positive_distances if d > t)
        # False Positives: أشخاص مختلفين، لكن مسافتهم أقل من العتبة
        fp = sum(1 for d in negative_distances if d < t)

        total_errors = fn + fp
        if total_errors < min_overlap:
            min_overlap = total_errors
            suggested_threshold = t

    print(f"\n🏆 Suggested Optimal Threshold: {suggested_threshold:.2f}")
    print(f"   (Use this value in your config.py)")


if __name__ == "__main__":
    evaluate_thresholds()