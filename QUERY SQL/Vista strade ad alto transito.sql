CREATE VIEW strade_alto_transito AS
SELECT 
    s.nome_strada,
    t.postazione,
    AVG(t.transiti_totali) AS transito_medio_giornaliero
FROM transiti t
JOIN strada s ON t.postazione = s.postazione
WHERE t.data <= '2019-12-24'
GROUP BY s.nome_strada, t.postazione
HAVING AVG(t.transiti_totali) > 10000;




