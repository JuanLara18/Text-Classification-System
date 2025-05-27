**************************************
* 0. Ajusta estas rutas y variables  *
**************************************
local fileA "/export/projects1/rsadun_daimler/05_Workplace/Clean Data/HR_monthly_panel.dta"
local fileB "/export/home/rcsguest/rcs_jcamacho/Projects/Class/Text-Classification-System/output/HR_monthly_panel_classified.dta"
local fileC "/export/projects1/rsadun_daimler/05_Workplace/Clean Data/HR_monthly_panel_translated.dta"

* Variables nuevas de B que queremos añadir a A
local newvars position_name_english position_category_gpt

**************************************
* 1. Prepara un temporal desde B     *
**************************************
tempfile Btemp
use "`fileB'", clear
gen long __obs = _n
keep __obs `newvars'
save "`Btemp'", replace

**************************************
* 2. Carga A y haz el merge final    *
**************************************
use "`fileA'", clear
gen long __obs = _n

* Merge 1:1 por fila, trae sólo las `newvars`
merge 1:1 __obs using "`Btemp'", keepusing(`newvars') nogen

* Ya no existe _merge, sólo quitamos __obs
drop __obs

**************************************
* 3. Guarda el resultado como C      *
**************************************
save "`fileC'", replace
