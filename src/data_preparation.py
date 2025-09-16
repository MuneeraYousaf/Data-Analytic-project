from datasets import load_dataset

# تحميل كامل البيانات (جميع الطرق الثلاث للتوليد)
dataset = load_dataset("KFUPM-JRCAI/arabic-generated-abstracts")

# يعرض أنواع التقسيمات الموجودة
print(dataset)

