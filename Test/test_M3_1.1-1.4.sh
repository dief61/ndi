#Test 1a – NORM mit Direktlookup (§-Referenz erkannt)
curl -s -X POST http://localhost:8000/api/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query":"Was steht in § 3 Abs. 1 NHundG?","debug":true}' \
  | python3 -m json.tool
 
 #Erwartet: query_typ: NORM, direktlookup: true, norm_reference: § 3 Abs. 1 NHundG, vektoren_anzahl: 2 (direct + step_back, kein HyDE)
 
########################################  
  #Test 1b – NORM ohne Direktlookup (HyDE + Step-Back aktiv):
  curl -s -X POST http://localhost:8000/api/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query":"Was muss ein Hundehalter nachweisen?","debug":true}' \
  | python3 -m json.tool
  
  # Erwartet: query_typ: NORM, direktlookup: false, vektoren_anzahl: 3 (direct + hyde + step_back)
  
###########################################
#Test 1c – IM-Query:  
curl -s -X POST http://localhost:8000/api/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query":"Welche Entitäten hat das Hunderegister?","debug":true}' \
  | python3 -m json.tool
  
#Erwartet: query_typ: IM, metadata_filter: {im_signals_exists: true}  

#Test 1d – ENTITY-Query:
curl -s -X POST http://localhost:8000/api/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query":"Wer ist zuständig für den Wesenstest?","debug":true}' \
  | python3 -m json.tool
  
#Erwartet: query_typ: ENTITY, metadata_filter: {norm_type: [COMPETENCE, DEF, MUST]}

#Test 1e – GENERAL-Query:
 curl -s -X POST http://localhost:8000/api/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query":"Hunderegister Niedersachsen","debug":true}' \
  | python3 -m json.tool
  
#Erwartet: query_typ: GENERAL, metadata_filter: {} (kein Filter)

#Schritt 1.2 – Retrieval-Klassen
#Test 2a – Nur Klasse A (Gesetzestext):
curl -s -X POST http://localhost:8000/api/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query":"§ 4 Kennzeichnung Transponder","debug":true}' \
  | python3 -m json.tool
  
#Erwartet: direktlookup: true (§-Referenz erkannt), aber chunks leer oder Fallback-Ergebnis (§ 99 existiert nicht)

#Schritt 1.3 – Parent-Child-Expansion
#Test 3a – Parent wird nachgeladen:
curl -s -X POST http://localhost:8000/api/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query":"theoretische Sachkundeprüfung Inhalte","debug":true}' \
  | python3 -m json.tool
  
#Erwartet: chunks_gesamt höher als chunks_angezeigt – Parent-Chunks wurden gefunden und sind in chunks_gesamt enthalten. In den Chunks: sowohl Child (hierarchy_level: 2) als auch Parent (hierarchy_level: 1) mit gleicher norm_reference.
  
  
#Schritt 1.4 – Cross-Reference-Expansion
#Test 4a – Querverweis-Auflösung:  
curl -s -X POST http://localhost:8000/api/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query":"gilt entsprechend § 5","debug":true}' \
  | python3 -m json.tool

# Erwartet: Chunk mit cross_references gesetzt + referenzierte Chunks ebenfalls in der Ergebnisliste (erkennbar an retrieval_source: crossref_tiefe_1)  
  
#Schnelltest aller Typen auf einmal
for query in \
  "Was steht in § 3 Abs. 1 NHundG?" \
  "Was muss ein Hundehalter nachweisen?" \
  "Wer ist zuständig für den Wesenstest?" \
  "Welche Entitäten hat das Hunderegister?" \
  "Hunderegister Niedersachsen"; do
  echo "─────────────────────────────────────"
  echo "Query: $query"
  curl -s -X POST http://localhost:8000/api/v1/rag/query \
    -H "Content-Type: application/json" \
    -d "{\"query\":\"$query\",\"debug\":true}" \
    | python3 -c "
import sys,json
d=json.load(sys.stdin)
print(f'  typ={d[\"query_typ\"]} direktlookup={d[\"direktlookup\"]} chunks={len(d[\"chunks\"])} vektoren={d[\"debug_info\"][\"vektoren_anzahl\"]}')
"
done





  
  
  
  
