#!/usr/bin/env python3
"""Analyze OPAL test results"""
import sys
sys.path.insert(0, '../stages/general')
import pandas as pd

df = pd.read_pickle('/media/sz/Data/Connected_Lecturers/Opal_test/raw/OPAL_ai_meta_test.p')

print("=" * 80)
print("OPAL Test - Neue Ergebnisse nach Verbesserungen")
print("=" * 80)
print(f"\nAnzahl Dokumente: {len(df)}")

print("\n" + "=" * 80)
print("AUTOREN-EXTRAKTION")
print("=" * 80)

for idx, row in df.iterrows():
    author = row.get('ai:author', '')
    revised = row.get('ai:revisedAuthor', '')
    affiliation = row.get('ai:affiliation', '')

    print(f"\n{idx+1}. Dokument:")
    print(f"   Raw Author:  '{author}'")
    print(f"   Revised:     {revised}")
    print(f"   Affiliation: {affiliation}")

# Statistiken
print("\n" + "=" * 80)
print("ZUSAMMENFASSUNG")
print("=" * 80)

total_docs = len(df)
empty_authors = (df['ai:author'] == '').sum() if 'ai:author' in df.columns else 0
filled_authors = total_docs - empty_authors

# Count revised authors (non-empty lists)
docs_with_revised = 0
if 'ai:revisedAuthor' in df.columns:
    for val in df['ai:revisedAuthor']:
        if val and str(val) != '[]' and str(val) != 'nan':
            docs_with_revised += 1

empty_affiliation = (df['ai:affiliation'] == '').sum() if 'ai:affiliation' in df.columns else 0
filled_affiliation = total_docs - empty_affiliation

print(f"\nAutoren:")
print(f"  ✓ Mit Autor extrahiert: {filled_authors}/{total_docs}")
print(f"  ✓ Leer (kein Autor):    {empty_authors}/{total_docs}")
print(f"  ✓ Mit validiertem Name: {docs_with_revised}/{total_docs}")

print(f"\nAffiliationen:")
print(f"  ✓ Mit Affiliation: {filled_affiliation}/{total_docs}")
print(f"  ✓ Leer:            {empty_affiliation}/{total_docs}")

# Prüfung: Institutionen als Autoren?
print("\n" + "=" * 80)
print("QUALITÄTSPRÜFUNG")
print("=" * 80)

institutional_keywords = ['tu dresden', 'htw', 'universität', 'hochschule', 'medienzentrum', 'institut']
problematic = []
unwanted_phrases = []

for idx, row in df.iterrows():
    author = str(row.get('ai:author', '')).lower()

    # Check for institutional names
    if author and any(keyword in author for keyword in institutional_keywords):
        problematic.append({
            'idx': idx + 1,
            'author': row.get('ai:author', '')
        })

    # Check for unwanted explanatory phrases
    if author and any(phrase in author for phrase in ['empty string', 'keine autoren', 'no author', '(', 'since']):
        unwanted_phrases.append({
            'idx': idx + 1,
            'author': row.get('ai:author', '')
        })

if problematic:
    print(f"\n⚠️  Institutionen als Autoren: {len(problematic)} Fälle")
    for p in problematic:
        print(f"   {p['idx']}. '{p['author']}'")
else:
    print("\n✅ Keine Institutionsnamen als Autoren gefunden!")

if unwanted_phrases:
    print(f"\n⚠️  Unerwünschte Phrasen: {len(unwanted_phrases)} Fälle")
    for p in unwanted_phrases:
        print(f"   {p['idx']}. '{p['author']}'")
else:
    print("\n✅ Keine unerwünschten Phrasen gefunden!")

print("\n" + "=" * 80)
print("VERBESSERUNG")
print("=" * 80)
print("\nVergleich vorher/nachher:")
print("  Vorher: 1x 'TU Dresden Medienzentrum' als Autor")
print("  Vorher: 3x unerwünschte Phrasen ('Empty string...', 'Keine Autoren...')")
print(f"  Nachher: {len(problematic)}x Institutionen, {len(unwanted_phrases)}x unerwünschte Phrasen")

if len(problematic) == 0 and len(unwanted_phrases) == 0:
    print("\n🎉 ERFOLG! Alle Probleme wurden behoben!")
else:
    print("\n📊 Noch vorhandene Probleme - weitere Verbesserungen nötig")
