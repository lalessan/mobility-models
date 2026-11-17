# Data: commuting between Danish municipalities (2024)

Source: Statistics Denmark, table PEND100, "Employed (end November) by municipality of
residence, area of workplace and sex", 2024. 98 municipalities (Christiansø left out).

## flows.csv
One row per pair of municipalities (98 × 98 = 9,604 rows, including i = j).

| column | meaning |
|---|---|
| origin_code, origin | municipality where people **live** |
| destination_code, destination | municipality where people **work** |
| commuters | number of employed people (men + women) |
| men, women | the same, by sex |

Rows with origin = destination are people working in their own municipality.

## municipalities.csv
| column | meaning |
|---|---|
| code, municipality | municipality code and name |
| workers_living | employed people who live here (row total of flows) |
| jobs | people who work here (column total of flows) |
| lon, lat | geographic centre of the municipality |

## distances.csv
Straight-line (great-circle) distance in km between the geographic centres of each pair
of different municipalities (98 × 97 = 9,506 rows).
