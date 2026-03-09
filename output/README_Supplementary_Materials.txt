
SUPPLEMENTARY MATERIALS SUMMARY
================================

TABLES CREATED:
───────────────
1. Table_1_Feature_Ranking_Week52.csv
   └─ Rankings of all 6 features by R² for primary target (FVC Week 52)
   
2. Table_2_Target_Summary_Stats.csv
   └─ Summary statistics (N, Mean, SD, Range) for all 4 target variables

FIGURES CREATED:
────────────────
1. Figure_1_R2_Heatmap.png
   └─ 6×4 heatmap showing R² for all feature-target combinations
   └─ Color scale: Red (negative) → Yellow (neutral) → Green (positive)
   └─ Key insight: Parenchymal features in green, airway features in red
   
2. Figure_2_Airway_vs_Parenchymal.png
   └─ Bar chart comparing mean R² of airway vs parenchymal categories
   └─ Shows stark separation across all targets
   └─ Parenchymal superior for week 0/52; both fail for decline
   
3. Figure_3_MAE_Distribution.png
   └─ Box plots of MAE across all features, grouped by target
   └─ Shows that cross-sectional targets (weeks) have higher MAE than decline
   └─ Highlights variability in prediction error across features

WHERE TO INSERT IN THESIS:
──────────────────────────
→ Figure 1 (Heatmap): New subsection "4.2 Visual Summary of Feature Performance"
  (before "Parenchymal Metrics Dominate")
  
→ Figure 2 & 3: Supplementary figures or Methods appendix
  (or reference in "Summary of Key Findings")

KEY INSIGHTS FROM SUPPLEMENTARY MATERIALS:
──────────────────────────────────────────
✓ Table 1 clearly shows R² ranking: histogram_entropy (0.408) >> others
✓ Figure 1 shows color pattern: bottom 2 green, top 4 red
✓ Figure 2 quantifies the gap: parenchymal mean R²=0.36, airway=-0.11 for Week52
✓ Figure 3 shows decline targets have inherently lower MAE but higher relative error

RECOMMENDATION:
───────────────
• Use Figure 1 (Heatmap) as PRIMARY visual summary early in results
• Tables are best for LaTeX appendix or supplementary materials
• Figures 2-3 provide supporting detail for decline prediction section
