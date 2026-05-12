SELECT
    nd.title,
    nd.source_type,
    MAX(nc.doc_class)                    AS klasse,
    COUNT(nc.id)                         AS chunks_gesamt,
    SUM(CASE WHEN nc.hierarchy_level = 1 THEN 1 ELSE 0 END) AS parent_chunks,
    SUM(CASE WHEN nc.hierarchy_level = 2 THEN 1 ELSE 0 END) AS child_chunks,
    SUM(nc.token_count)                  AS tokens_gesamt,
    ROUND(AVG(nc.token_count))           AS avg_tokens,
    MIN(nc.token_count)                  AS min_tokens,
    MAX(nc.token_count)                  AS max_tokens
FROM norm_documents nd
LEFT JOIN norm_chunks nc ON nc.doc_id = nd.id
GROUP BY nd.title, nd.source_type
ORDER BY chunks_gesamt DESC;