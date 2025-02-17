CREATE VIEW bo_view AS
SELECT *
FROM mts_anagrafica
WHERE postazione IN (
    SELECT postazione 
    FROM postazioni 
    WHERE provincia = 'BO'
)
AND data <= '2019-12-24';

CREATE VIEW mo_view AS
SELECT *
FROM mts_anagrafica
WHERE postazione IN (
    SELECT postazione 
    FROM postazioni 
    WHERE provincia = 'MO'
)
AND data <= '2019-12-24';

CREATE VIEW re_view AS
SELECT *
FROM mts_anagrafica
WHERE postazione IN (
    SELECT postazione 
    FROM postazioni 
    WHERE provincia = 'RE'
)
AND data <= '2019-12-24';

CREATE VIEW pr_view AS
SELECT *
FROM mts_anagrafica
WHERE postazione IN (
    SELECT postazione 
    FROM postazioni 
    WHERE provincia = 'PR'
)
AND data <= '2019-12-24';

CREATE VIEW pc_view AS
SELECT *
FROM mts_anagrafica
WHERE postazione IN (
    SELECT postazione 
    FROM postazioni 
    WHERE provincia = 'PC'
)
AND data <= '2019-12-24';

CREATE VIEW fe_view AS
SELECT *
FROM mts_anagrafica
WHERE postazione IN (
    SELECT postazione 
    FROM postazioni 
    WHERE provincia = 'FE'
)
AND data <= '2019-12-24';

CREATE VIEW rn_view AS
SELECT *
FROM mts_anagrafica
WHERE postazione IN (
    SELECT postazione 
    FROM postazioni 
    WHERE provincia = 'RN'
)
AND data <= '2019-12-24';

CREATE VIEW fc_view AS
SELECT *
FROM mts_anagrafica
WHERE postazione IN (
    SELECT postazione 
    FROM postazioni 
    WHERE provincia = 'FC'
)
AND data <= '2019-12-24';




