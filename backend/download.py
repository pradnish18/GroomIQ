from icrawler.builtin import BingImageCrawler
import os

# ===== ALL 11 CLASSES WITH MULTIPLE SPECIFIC KEYWORDS =====
# Using varied, specific keywords per class so Bing returns
# different images each run — avoids duplicates being skipped

classes = {

    "dry": [
        "dry damaged brittle hair split ends",
        "dehydrated dull dry hair texture closeup",
        "dry hair before after moisture treatment",
    ],

    "frizzy": [
        "frizzy puffy uncontrolled hair humidity",
        "frizzy hair texture closeup volume",
        "anti frizz hair problem before after",
    ],

    "hairfall": [
        "hair loss thinning scalp alopecia",
        "hair fall problem receding hairline woman",
        "thinning hair loss bald patch scalp",
    ],

    "Straight": [
        "straight sleek shiny hair woman",
        "straight hair texture flat iron result",
        "straight black asian hair closeup",
    ],

    "Wavy": [
        "wavy beach wave hair texture",
        "natural wavy hair S wave pattern",
        "wavy hair woman medium length",
    ],

    "bald": [
        "bald head man shaved smooth scalp",
        "male pattern baldness bald scalp top view",
        "bald woman alopecia areata no hair",
    ],

    "curly": [
        "curly hair coils defined curls woman",
        "3c curly hair texture ringlets",
        "natural curly hair afro texture",
    ],

    "dreadlocks": [
        "dreadlocks locs hairstyle closeup",
        "mature dreadlocks loc hair texture",
        "thin thick dreadlocks black hair",
    ],

    "healthy": [
        "healthy shiny strong hair woman",
        "glossy healthy hair texture shine",
        "healthy hair after treatment before after",
    ],

    "kinky": [
        "kinky coily 4c hair texture afro",
        "tight coils kinky natural hair shrinkage",
        "4c kinky hair natural black woman",
    ],

    "notbald": [
        "full thick hair density scalp coverage",
        "hair density normal healthy scalp",
        "thick full hair growth scalp top view",
    ],
}

# ===== TARGET COUNT PER CLASS =====
TARGET = 600

# ===== DOWNLOAD =====
dataset_path = "../datasets"

for folder, keywords in classes.items():
    class_path = os.path.join(dataset_path, folder)

    # Count existing images
    existing = 0
    if os.path.exists(class_path):
        existing = len([
            f for f in os.listdir(class_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

    print(f"\n{'='*50}")
    print(f"📁 {folder}: {existing} images (target: {TARGET})")

    if existing >= TARGET:
        print(f"✅ Already at target — skipping")
        continue

    # Download with each keyword variant to get diverse images
    for keyword in keywords:
        # Recount before each keyword batch
        current = len([
            f for f in os.listdir(class_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]) if os.path.exists(class_path) else 0

        if current >= TARGET:
            print(f"  ✅ Reached target mid-download — stopping")
            break

        print(f"  🔍 Keyword: '{keyword}'")
        crawler = BingImageCrawler(
            storage={'root_dir': class_path},
            downloader_threads=4,
        )
        crawler.crawl(
            keyword=keyword,
            max_num=200,   # 200 per keyword × 3 keywords = up to 600 attempts
            min_size=(100, 100),  # skip tiny/icon images
        )

    # Final count
    final = len([
        f for f in os.listdir(class_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    print(f"  📊 Final count: {final} images")

print("\n🎉 ALL DOWNLOADS COMPLETE!")
print("\nFinal counts:")
for folder in classes:
    path = os.path.join(dataset_path, folder)
    if os.path.exists(path):
        count = len([f for f in os.listdir(path) if f.lower().endswith(('.jpg','.jpeg','.png'))])
        status = "✅" if count >= TARGET else "⚠️ "
        print(f"  {status} {folder}: {count} images")