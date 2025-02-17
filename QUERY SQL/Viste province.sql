CREATE VIEW bo_view AS
SELECT *
FROM mts_clean
WHERE provincia = 'BO'
AND giorno <= '2019-12-24';

CREATE VIEW mo_view AS
SELECT *
FROM mts_clean
WHERE provincia = 'MO'
AND giorno <= '2019-12-24';

CREATE VIEW re_view AS
SELECT *
FROM mts_clean
WHERE provincia = 'RE'
AND giorno <= '2019-12-24';

CREATE VIEW pr_view AS
SELECT *
FROM mts_clean
WHERE provincia = 'PR'
AND giorno <= '2019-12-24';

CREATE VIEW pc_view AS
SELECT *
FROM mts_clean
WHERE provincia = 'PC'
AND giorno <= '2019-12-24';

CREATE VIEW fe_view AS
SELECT *
FROM mts_clean
WHERE provincia = 'FE'
AND giorno <= '2019-12-24';

CREATE VIEW rn_view AS
SELECT *
FROM mts_clean
WHERE provincia = 'RN'
AND giorno <= '2019-12-24';

CREATE VIEW fc_view AS
SELECT *
FROM mts_clean
WHERE provincia = 'FC'
AND giorno <= '2019-12-24';




