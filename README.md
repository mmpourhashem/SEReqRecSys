# 📌 Adaptive Hybrid Recommender System for Requirements Reuse

This project implements a hybrid recommender system tailored for software requirements engineering. Designed for seamless integration into existing workflows, it treats stakeholders as users and requirements as recommendable items. Once a stakeholder interacts with at least two requirements, the system generates personalized predictions to suggest the most relevant items based on inferred preferences.

---

## 🔧 Key Features

- Personalized requirement recommendations using collaborative and content-based filtering  
- Lightweight integration with minimal disruption to engineering processes  
- Offline semantic similarity computation using GloVe vectors  
- Domain-specific TF-IDF scoring based on the PURE corpus  
- Scalable and sustainable update cycle aligned with engineering iterations  

---

## 🛠️ Maintenance Workflow

- Regenerate semantic similarity matrix post each engineering cycle  
- Recalculate TF-IDF scores to reflect evolving project contexts  
- All updates are computationally efficient and infrastructure-friendly  

---

This system is ideal for teams aiming to enhance requirement reuse and stakeholder alignment in modern software development.
