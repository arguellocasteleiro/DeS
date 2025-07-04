# -*- coding: utf-8 -*-
"""pub-zoo_1lineCode.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/124...
"""

# ___________________________ requirements
### please, download MRCONSO.RRF from
### https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html

###--- --- --- --- --- --- --- python libraries
import re, string, os, sys
import datetime
#--- --- --- Google Colab
from google.colab import output, runtime, drive

# ___________________________ *** path to folders [ path to file MRCONSO.RRF ]
### please, store the file MRCONSO.RRF within the folder
EXP_dir = "/content/drive/MyDrive/pub-zoo_files/"

if (EXP_dir.startswith("/content/drive")):
  drive.mount('/content/drive')

# ______________________________________________________ FHIR ValueSet Instances
# ______________________________________________________ extensional value sets
# ______________________________________________________ format: 'JSON', 'RDF', or non-normative 'JSON-LD'
# ___________________________
strTemp0 = '{ "resourceType" : "ValueSet",' + "\n"
strTemp0 += '  "id" : "<vsName>",' + "\n"
strTemp0 += '  "status" : "draft",' + "\n"
strTemp0 += '  "compose" : {' + "\n"
strTemp0 += '  "lockedDate" : "<YYYY-MM-DD>",' + "\n"
strTemp0 += '  "include" : [' + "\n"
strTemp0 += '<<system, concept(code, display)>>' + "\n"
strTemp0 += ' ] }}' + "\n"
# ___________________________
strTemp1 = '@prefix fhir: <http://hl7.org/fhir/> .' + "\n"
strTemp1 += '@prefix owl: <http://www.w3.org/2002/07/owl#> .' + "\n"
strTemp1 += '@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .' + "\n"
strTemp1 += '@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .' + "\n"
strTemp1 += '@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .' + "\n"
strTemp1 += '# - resource --------------------------------------------' + "\n"
strTemp1 += '[ a fhir:ValueSet ;' + "\n"
strTemp1 += '  fhir:nodeRole fhir:treeRoot ;' + "\n"
strTemp1 += '  fhir:id [ fhir:v "<vsName>"] ; # ' + "\n"
strTemp1 += '  fhir:status [ fhir:v "draft"] ; # ' + "\n"
strTemp1 += '  fhir:compose [' + "\n"
strTemp1 += '     fhir:lockedDate [ fhir:v "<YYYY-MM-DD>"^^xsd:date ] ;' + "\n"
strTemp1 += '     fhir:include ([ ' + "\n"
strTemp1 += '<<system, concept(code, display)>>' + "\n"
strTemp1 += '     )] ' + "\n"
strTemp1 += '] . # ' + "\n"
strTemp1 += '#--------------------------------------------' + "\n"
# ___________________________
strTemp3 = '{ "@context": {' + "\n"
strTemp3 += '         "xsd": "http://www.w3.org/2001/XMLSchema#",' + "\n"
strTemp3 += '         "rdfs": "http://www.w3.org/2000/01/rdf-schema#",' + "\n"
strTemp3 += '         "owl": "http://www.w3.org/2002/07/owl#",' + "\n"
strTemp3 += '         "fhir": "http://hl7.org/fhir/",' + "\n"
strTemp3 += '          "pz": "http://pub-zoo.org/"},' + "\n"
strTemp3 += ' "@graph": [{"@id": "pz:<vsName>/ValueSet", ' + "\n"
strTemp3 += '     "@type": "fhir:ValueSet",' + "\n"
strTemp3 += '     "fhir:nodeRole": {"@id": "fhir:treeRoot"},' + "\n"
strTemp3 += '     "fhir:status": {"@value": "draft", "@type": "fhir:value"},' + "\n"
strTemp3 += '     "fhir:compose": {"@id": "pz:<vsName>/extensional"}' + "\n"
strTemp3 += '     },{"@id": "pz:<vsName>/extensional",' + "\n"
strTemp3 += '       "fhir:lockedDate": {"@value": "<YYYY-MM-DD>", "@type": "xsd:date" },' + "\n"
strTemp3 += '        "fhir:include":'
strTemp3 += '<<system, concept(code, display)>>'
strTemp3 += ' }] }' + "\n"
# ___________________________
# ______________________________________________________ nanoPublications
# ______________________________________________________ format: JSON-LD
# ___________________________
strTemp4 = '{    ' + "\n"
strTemp4 += '   "@context": {' + "\n"
strTemp4 += ' 	"np": "http://www.nanopub.org/nschema#",' + "\n"
strTemp4 += ' 	"dct": "http://purl.org/dc/terms/",' + "\n"
strTemp4 += ' 	"xsd": "http://www.w3.org/2001/XMLSchema#",' + "\n"
strTemp4 += ' 	"rdfs": "http://www.w3.org/2000/01/rdf-schema#",' + "\n"
strTemp4 += ' 	"owl": "http://www.w3.org/2002/07/owl#",' + "\n"
strTemp4 += ' 	"skos": "http://www.w3.org/2004/02/skos/core#",' + "\n"
strTemp4 += ' 	"prov": "http://www.w3.org/ns/prov-o#",' + "\n"
strTemp4 += ' 	"obo": "http://purl.obolibrary.org/obo/",' + "\n"
strTemp4 += ' 	"sio": "http://semanticscience.org/resource/",' + "\n"
strTemp4 += ' 	"umls": "https://uts.nlm.nih.gov/uts/umls",' + "\n"
strTemp4 += ' 	"orcid": "http://orcid.org/",' + "\n"
strTemp4 += ' 	"pz": "http://pub-zoo.org/"' + "\n"
strTemp4 += '     },' + "\n"
strTemp4 += '     "@graph": [{' + "\n"
strTemp4 += ' 	"@id": "pz:pub<yearQ>/<disease_Name>/Head",' + "\n"
strTemp4 += ' 	 "@graph": [{"@id": "pz:pub<yearQ>/<disease_Name>/",' + "\n"
strTemp4 += ' 	"np:hasPublicationInfo": {"@id": "pz:pub<yearQ>/pubinfo"},' + "\n"
strTemp4 += ' 	"np:hasProvenance": {"@id": "pz:pub<yearQ>/<disease_Name>/provenance"},' + "\n"
strTemp4 += ' 	"np:hasAssertion": { "@id": "pz:pub<yearQ>/<disease_Name>/assertion"},' + "\n"
strTemp4 += ' 	"@type": "np:Nanopublication"}]' + "\n"
strTemp4 += ' 	},{' + "\n"
strTemp4 += ' 	"@id": "pz:pub<yearQ>/<disease_Name>/assertion",' + "\n"
strTemp4 += ' 	"@graph": [{"@id": "umls:<disease_CUI>","sio:SIO_001403": {"@id": "umls:<related_CUI>"} },' + "\n"
strTemp4 += ' 	{"@id": "umls:<disease_CUI>","skos:hiddenLabel": [<<terms/expressions>>] }]' + "\n"
strTemp4 += '       },{' + "\n"
strTemp4 += ' 	    "@id": "pz:pub<yearQ>/<disease_Name>/provenance",' + "\n"
strTemp4 += '             "@graph": [{ "@id": "pz:pub<yearQ>/<disease_Name>/assertion",' + "\n"
strTemp4 += '             "prov:value": { "@value": "<paragraph>" ,"@type": "xsd:string" },' + "\n"
strTemp4 += '             "prov:wasQuotedFrom": {"@id": "pz:pub<yearQ>/pubinfo"},' + "\n"
strTemp4 += ' 	     "obo:RO_0002558": {"@id": "obo:ECO_0000218"} }]' + "\n"
strTemp4 += '     },{' + "\n"
strTemp4 += ' 	"@id": "pz:pub<yearQ>/pubinfo",' + "\n"
strTemp4 += ' 	"@graph": [{"@id": "pz:pub<yearQ>/",' + "\n"
strTemp4 += '  "dct:creator": {"@id": "orcid:<ORCID>"},' + "\n"
strTemp4 += '  "dct:created": {"@value": "<YYYY-MM-DD>T10:00:00Z","@type": "xsd:dateTime" },' + "\n"
strTemp4 += ' 	"dct:title": {"@value": "<pub_Title>","@type": "xsd:string" },' + "\n"
strTemp4 += ' 	"dct:issued": {"@value": "<pub_DateIssued>","@type": "xsd:gYearMonth" },' + "\n"
strTemp4 += ' 	"dct:identifier": {"@id": "<pub_Identifier>"} }] ' + "\n"
strTemp4 += ' }]}' + "\n"
# ___________________________
# ______________________________________________________ MRCONSO.RRF
# ______________________________________________________ get the structured coded-data
# ___________________________ *** read MRCONSO.RRF
# ___________________________ *** and get Only codes for CUIs
def RgetMRCONSOcodesSCT(fin,finArr,finLang,finVOC,fno):
  tmp_data1 = '###' + '###'.join(finArr) + '###'
  tmp_l2 = []
  tmp_txt = ''
  tmp_data = ''
  tmp_lt = []
  tmp_str = ''
  tmp_cntTOT = 0
  tmp_cntTOT1 = 0

  with open(os.path.expanduser(fin),"r+") as tmp_f:
    for line in tmp_f:
      tmp_txt = line
      tmp_txt = tmp_txt.strip()
      if (tmp_txt.find('|') > -1):
        tmp_cntTOT += 1
        tmp_l2 = tmp_txt.split('|')
        #--- print tmp_l2[0] + '... ... ...' + tmp_l2[4]
        if (tmp_data1.find('###' + tmp_l2[0] + '###') > -1):
          if (tmp_l2[1] == finLang and tmp_l2[11] == finVOC):
            tmp_cntTOT1 += 1
            if (tmp_str == ''):
              tmp_str = '###' +  tmp_l2[13] + '###'
              tmp_data = "'" + tmp_l2[13] + "'"
            else:
              if (tmp_str.find('###' +  tmp_l2[13] + '###') == -1):
                tmp_str += tmp_l2[13] + '###'
                tmp_data += ",'" + tmp_l2[13] + "'"

  tmp_f.close()
  #--- --- --- --- ---
  print('matched...' + str(tmp_cntTOT1))
  tmp_data = '[' + tmp_data + ']'
  tmp_f1=open(os.path.expanduser(fno), "w")
  tmp_f1.write(tmp_data + "\n")
  print(tmp_data)
  tmp_f1.close()
#-------------------------------------EndFunction
# ___________________________ *** read MRCONSO.RRF
# ___________________________ *** and get information about CUIs
def RgetMRCONSOInfo(fin,finArr,finLang,finVOC,fno):
  tmp_data1 = '###' + '###'.join(finArr) + '###'
  tmp_l2 = []
  tmp_txt = ''
  tmp_data = ''
  tmp_lt = []
  tmp_str = ''
  tmp_cntTOT = 0
  tmp_cntTOT1 = 0
  tmp_eka = 0

  tmp_f1=open(os.path.expanduser(fno), "w")
  with open(os.path.expanduser(fin),"r+") as tmp_f:
    for line in tmp_f:
      tmp_txt = line
      tmp_txt = tmp_txt.strip()
      if (tmp_txt.find('|') > -1):
        tmp_cntTOT += 1
        tmp_l2 = tmp_txt.split('|')
        #--- print tmp_l2[0] + '... ... ...' + tmp_l2[4]
        if (tmp_data1.find('###' + tmp_l2[0] + '###') > -1):
            tmp_eka = 0
            if (finVOC == '' and tmp_l2[1] == finLang):
                        tmp_eka = 1
            if (tmp_l2[1] == finLang and tmp_l2[11] == finVOC and tmp_l2[12] == 'FN'):
                        tmp_eka = 1
            if (tmp_eka == 1):
                        tmp_cntTOT1 += 1
                        tmp_data = tmp_l2[0] + '###' + tmp_l2[11] + '###' + tmp_l2[13] + '###' + tmp_l2[14] + '###'
                        tmp_f1.write(tmp_data + "\n")
  #--- --- --- --- ---
  print('matched...' + str(tmp_cntTOT1))
  tmp_f1.close()
  tmp_f.close()
#-------------------------------------EndFunction
# ______________________________________________________ JSON, RDF Turtle, JSON-LD
# ______________________________________________________ FHIR ValueSet Instances
def RcreateFHIRvs(fnFormat,fin,fnNameVS):
	tmp_eka = 0
	if (fnFormat == 'JSON'):
		RpopTemp0(fin,EXP_dir + fnNameVS + ".json", fnNameVS)
		tmp_eka = 1
	if (fnFormat == 'RDF'):
		RpopTemp1(fin,EXP_dir + fnNameVS + ".ttl", fnNameVS)
		tmp_eka = 1
	if (fnFormat == 'JSON-LD'):
		RpopTemp3(fin,EXP_dir + fnNameVS + ".jsonld", fnNameVS)
		tmp_eka = 1
	if (tmp_eka == 0):
		print("something went very wrong! the formats are: 'JSON', 'RDF', or non-normative 'JSON-LD'")
	else:
		print("all done! have a nice day! :-))")
#-------------------------------------EndFunction
def RpopTemp0(fin,fno,fnStr):
	tmp_f = open(os.path.expanduser(fin),"r+")
	tmp_data = tmp_f.read()
	tmp_f.close()

	tmp_l = tmp_data.split("\n")
	tmp_txt = ''
	tmp_data = ''
	tmp_data1 = ''
	tmp_data2 = ''
	tmp_data3 = ''
	tmp_lt = []
	tmp_l1 = []
	tmp_str = ''
	tmp_txtA = strTemp0
	#--- --- --- --- ---
	strDateCreated = datetime.datetime.now()
	tmp_str = str(strDateCreated.year) + '-' + str(strDateCreated.strftime("%m")) + '-' + str(strDateCreated.strftime("%d"))
	tmp_txtA = tmp_txtA.replace('<YYYY-MM-DD>',tmp_str)
	tmp_txtA = tmp_txtA.replace('<vsName>',fnStr)
  #--- --- --- --- ---
	tmp_cnt = 0
	while tmp_cnt < len(tmp_l):
		tmp_txt = tmp_l[tmp_cnt].strip()
		if (tmp_txt.find('###') > -1):
			tmp_lt = tmp_txt.split('###')
			if (tmp_lt[1] == 'SNOMEDCT_US'):
				if (tmp_data1 != ''):
					tmp_data1 += ',' + "\n"
				tmp_data1 += "\t" + '{"code" : "' + tmp_lt[2] + '", "display" : "' + tmp_lt[3]  + '"}'
			if (tmp_lt[1] == 'SNOMEDCT_VET'):
				if (tmp_data2 != ''):
					tmp_data2 += ',' + "\n"
				tmp_data2 += "\t" + '{"code" : "' + tmp_lt[2] + '", "display" : "' + tmp_lt[3]  + '"}'
			if (tmp_lt[1] != 'SNOMEDCT_US' and tmp_lt[1] != 'SNOMEDCT_VET'):
				if (tmp_data3 != ''):
					tmp_data3 += ',' + "\n"
				tmp_data3 += "\t" + '{"code" : "not-known-code", "display" : "' + tmp_lt[3]  + '"}'
		#---
		tmp_cnt = tmp_cnt + 1
	#--- --- --- --- ---
	if (tmp_data1 != ''):
		tmp_data1 = "\t" + '"system" : "http://snomed.info/sct", ' + "\n" + "\t" + '"concept": [' + "\n" + tmp_data1 + ']' + "\n"
		tmp_l1.append(tmp_data1)
	if (tmp_data2 != ''):
		tmp_data2 = "\t" + '"system" : "https://vtsl.vetmed.vt.edu/vetsct", ' + "\n" + "\t" + '"concept": [' + "\n" + tmp_data2 + ']' + "\n"
		tmp_l1.append(tmp_data2)
	if (tmp_data3 != ''):
		tmp_data3 = "\t" + '"system" : "http://not-a-known-code-system", ' + "\n" + "\t" + '"concept": [' + "\n" +  tmp_data3 + ']' + "\n"
		tmp_l1.append(tmp_data3)
	#--- --- --- --- ---
	tmp_str = "\t" + '},{' + "\n"
	tmp_data = "\t" + '{' +  "\n" + tmp_str.join(tmp_l1) + "\t" + '}'
	tmp_f1=open(os.path.expanduser(fno), "w")
	tmp_txtA = tmp_txtA.replace('<<system, concept(code, display)>>',tmp_data)
	tmp_f1.write("\n" + tmp_txtA + "\n")
	tmp_f1.close()
  #--- --- --- --- ---
	print('--- --- --- --- --- --- ---')
	print(tmp_txtA)
	print('--- --- --- --- --- --- ---')
#-------------------------------------EndFunction
def RpopTemp1(fin,fno,fnStr):
	tmp_f = open(os.path.expanduser(fin),"r+")
	tmp_data = tmp_f.read()
	tmp_f.close()

	tmp_l = tmp_data.split("\n")
	tmp_txt = ''
	tmp_data = ''
	tmp_data1 = ''
	tmp_data2 = ''
	tmp_data3 = ''
	tmp_lt = []
	tmp_l1 = []
	tmp_str = ''
	tmp_txtA = strTemp1
	#--- --- --- --- ---
	strDateCreated = datetime.datetime.now()
	tmp_str = str(strDateCreated.year) + '-' + str(strDateCreated.strftime("%m")) + '-' + str(strDateCreated.strftime("%d"))
	tmp_txtA = tmp_txtA.replace('<YYYY-MM-DD>',tmp_str)
	tmp_txtA = tmp_txtA.replace('<vsName>',fnStr)
	#--- --- --- --- ---
	tmp_cnt = 0
	while tmp_cnt < len(tmp_l):
		tmp_txt = tmp_l[tmp_cnt].strip()
		if (tmp_txt.find('###') > -1):
			tmp_lt = tmp_txt.split('###')
			if (tmp_lt[1] == 'SNOMEDCT_US'):
				if (tmp_data1 == ''):
					tmp_data1 +=  "\t" + 'fhir:concept ([' + "\n"
				else:
					tmp_data1 +=  "\t" + '][' + "\n"
				tmp_data1 +=  "\t" +"\t" + 'fhir:code [ fhir:v "'+ tmp_lt[2] + '" ] ;' + "\n"
				tmp_data1 += "\t" + "\t" + 'fhir:display [ fhir:v "'+ tmp_lt[3] + '" ]' + "\n"
			if (tmp_lt[1] == 'SNOMEDCT_VET'):
				if (tmp_data2 == ''):
					tmp_data2 += "\t" + 'fhir:concept ([' + "\n"
				else:
					tmp_data2 +=  "\t" + '][' + "\n"
				tmp_data2 += "\t" + "\t" + 'fhir:code [ fhir:v "'+ tmp_lt[2] + '" ] ;' + "\n"
				tmp_data2 += "\t" + "\t" + 'fhir:display [ fhir:v "'+ tmp_lt[3] + '" ]' + "\n"
			if (tmp_lt[1] != 'SNOMEDCT_US' and tmp_lt[1] != 'SNOMEDCT_VET'):
				if (tmp_data3 == ''):
					tmp_data3 += "\t" + 'fhir:concept ([' + "\n"
				else:
					tmp_data3 +=  "\t" + '][' + "\n"
				tmp_data3 += "\t" + "\t" + 'fhir:code [ fhir:v "not-known-code" ] ;' + "\n"
				tmp_data3 += "\t" + "\t" + 'fhir:display [ fhir:v "'+ tmp_lt[3] + '" ]' + "\n"
		#---
		tmp_cnt = tmp_cnt + 1
	#--- --- --- --- ---
	if (tmp_data1 != ''):
		tmp_data1 = "\t" + 'fhir:system [ fhir:v "http://snomed.info/sct"^^xsd:anyURI ] ;' + "\n" + tmp_data1 + "\t" + ' ])' + "\n"
		tmp_l1.append(tmp_data1)
	if (tmp_data2 != ''):
		tmp_data2 = "\t" + 'fhir:system [ fhir:v "https://vtsl.vetmed.vt.edu/vetsct"^^xsd:anyURI ] ;' + "\n" + tmp_data2 + "\t" + ' ])' + "\n"
		tmp_l1.append(tmp_data2)
	if (tmp_data3 != ''):
		tmp_data3 = "\t" + 'fhir:system [ fhir:v "http://not-a-known-code-system"^^xsd:anyURI ] ;' + "\n" + tmp_data3 + "\t" + ' ])' + "\n"
		tmp_l1.append(tmp_data3)
	#--- --- --- --- ---
	tmp_str = "\t" + '] [' + "\n"
	tmp_data = tmp_str.join(tmp_l1) + "\t" + ' ]'
	tmp_f1=open(os.path.expanduser(fno), "w")
	tmp_txtA = tmp_txtA.replace('<<system, concept(code, display)>>',tmp_data)
	tmp_f1.write("\n" + tmp_txtA + "\n")
	tmp_f1.close()
	print('--- --- --- --- --- --- ---')
	print(tmp_txtA)
	print('--- --- --- --- --- --- ---')
#-------------------------------------EndFunction
def RpopTemp3(fin,fno,fnStr):
	tmp_f = open(os.path.expanduser(fin),"r+")
	tmp_data = tmp_f.read()
	tmp_f.close()

	tmp_l = tmp_data.split("\n")
	tmp_txt = ''
	tmp_data = ''
	tmp_data1 = ''
	tmp_data2 = ''
	tmp_data3 = ''
	tmp_lt = []
	tmp_l1 = []
	tmp_l3 = []
	tmp_str = ''
	tmp_txtA = strTemp3
	#--- --- --- --- ---
	strDateCreated = datetime.datetime.now()
	tmp_str = str(strDateCreated.year) + '-' + str(strDateCreated.strftime("%m")) + '-' + str(strDateCreated.strftime("%d"))
	tmp_txtA =  tmp_txtA.replace('<YYYY-MM-DD>',tmp_str)
	#--- --- --- --- ---
	tmp_cnt = 0
	while tmp_cnt < len(tmp_l):
		tmp_txt = tmp_l[tmp_cnt].strip()
		if (tmp_txt.find('###') > -1):
			tmp_lt = tmp_txt.split('###')
			if (tmp_lt[1] == 'SNOMEDCT_US'):
				if (tmp_data1 == ''):
					tmp_l3.append('"@id": "pz:<vsName>/SCT"')
					tmp_data1 = "\t" + ' "@id": "pz:<vsName>/SCT",' + "\n"
					tmp_data1 += "\t" + '"fhir:system":{"@value":"http://snomed.info/sct", "@type": "xsd:anyURI" },' + "\n"
					tmp_data1 += "\t" + '"@graph": [' + "\n" + "\t" + '{'
				else:
					tmp_data1 += "\t"  + '},{' + "\n" + "\t"
				tmp_data1 += '"@id": "http://snomed.info/sct/' + tmp_lt[2] + '",' + "\n"
				tmp_data1 += "\t" + '"@type": "fhir:concept",' + "\n"
				tmp_data1 += "\t" + '"fhir:code":{"@value":"' + tmp_lt[2]  + '", "@type": "fhir:value" },' + "\n"
				tmp_data1 += "\t" + '"fhir:display":{"@value":"' + tmp_lt[3] + '", "@type": "fhir:value" }' + "\n"
			if (tmp_lt[1] == 'SNOMEDCT_VET'):
				if (tmp_data2 == ''):
					tmp_l3.append('"@id": "pz:<vsName>/VetSCT"')
					tmp_data2 = "\t" + ' "@id": "pz:<vsName>/VetSCT",' + "\n"
					tmp_data2 += "\t" + '"fhir:system":{"@value":"https://vtsl.vetmed.vt.edu/vetsct", "@type": "xsd:anyURI" },' + "\n"
					tmp_data2 += "\t" + '"@graph": [' + "\n" + "\t" + '{'
				else:
					tmp_data2 += "\t"  + '},{' + "\n" + "\t"
				tmp_data2 += '"@id": "https://vtsl.vetmed.vt.edu/vetsct/' + tmp_lt[2] + '",' + "\n"
				tmp_data1 += "\t" + '"@type": "fhir:concept",' + "\n"
				tmp_data2 += "\t" + '"fhir:code":{"@value":"' + tmp_lt[2]  + '", "@type": "fhir:value" },' + "\n"
				tmp_data2 += "\t" + '"fhir:display":{"@value":"' + tmp_lt[3] + '", "@type": "fhir:value" }' + "\n"
			if (tmp_lt[1] != 'SNOMEDCT_US' and tmp_lt[1] != 'SNOMEDCT_VET'):
				if (tmp_data3 == ''):
					tmp_l3.append('"@id": "pz:<vsName>/unknown"')
					tmp_data3 = "\t" + ' "@id": "pz:<vsName>/unknown",' + "\n"
					tmp_data3 += "\t" + '"fhir:system":{"@value":"http://not-a-known-code-system", "@type": "xsd:anyURI" },' + "\n"
					tmp_data3 += "\t" + '"@graph": [' + "\n" + "\t" + '{'
				else:
					tmp_data3 += "\t"  + '},{' + "\n" + "\t"
				tmp_data3 += '"@id": "pz:' + fnStr + '/not-known-code_' + tmp_lt[3].replace(' ', '_') + '",' + "\n"
				tmp_data3 += "\t" + '"@type": "fhir:concept",' + "\n"
				tmp_data3 += "\t" + '"fhir:code":{"@value":"not-known-code", "@type": "fhir:value" },' + "\n"
				tmp_data3 += "\t" + '"fhir:display":{"@value":"' + tmp_lt[3] + '", "@type": "fhir:value" }' + "\n"
		#---
		tmp_cnt = tmp_cnt + 1
	#--- --- --- --- ---
	tmp_str = '[{' + '},{'.join(tmp_l3) + '}]' + "\n"
	tmp_data = tmp_str + "\t" + ' },{' + "\n"
	#--- --- --- --- ---
	if (tmp_data1 != ''):
		tmp_data1 += "\t" + '}]' + "\n"
		tmp_l1.append(tmp_data1)
	if (tmp_data2 != ''):
		tmp_data2 += "\t" + '}]' + "\n"
		tmp_l1.append(tmp_data2)
	if (tmp_data3 != ''):
		tmp_data3 += "\t" + '}]' + "\n"
		tmp_l1.append(tmp_data3)
	#--- --- --- --- ---
	tmp_str = "\t" + ' },{' + "\n"
	tmp_data += tmp_str.join(tmp_l1)
	#--- --- --- --- ---
	tmp_f1=open(os.path.expanduser(fno), "w")
	tmp_txtA = tmp_txtA.replace('<<system, concept(code, display)>>',tmp_data)
	tmp_txtA = tmp_txtA.replace('<vsName>',fnStr)
	tmp_f1.write("\n" + tmp_txtA + "\n")
	tmp_f1.close()
#-------------------------------------EndFunction
# ______________________________________________________ JSON-LD
# ______________________________________________________ nanoPublications
def RcreateNanoPub(fin,strORCID):
	tmp_f = open(os.path.expanduser(fin),"r+")
	tmp_data = tmp_f.read()
	tmp_f.close()

	fno = fin.replace('.txt', '_np.jsonld')

	tmp_l = tmp_data.split("\n")
	tmp_txt = ''
	tmp_data = ''
	tmp_lt = []
	tmp_str = ''
	tmp_cntTOT = 0
	tmp_txtA = strTemp4
	#--- --- --- --- ---
	strDateCreated = datetime.datetime.now()
	tmp_str = str(strDateCreated.year) + '-' + str(strDateCreated.strftime("%m")) + '-' + str(strDateCreated.strftime("%d"))
	tmp_txtA = tmp_txtA.replace('<YYYY-MM-DD>',tmp_str)
	#--- --- --- --- ---
	tmp_txtA = tmp_txtA.replace('<ORCID>',strORCID)
	#--- --- --- --- ---
	#--- --- --- --- --- read line
	tmp_cnt = 0
	while tmp_cnt < len(tmp_l):
		tmp_txt = tmp_l[tmp_cnt].strip()
		if (tmp_txt.find('###') > -1):
			tmp_lt = tmp_txt.split('###')
			tmp_cntTOT += 1
			tmp_txtA = tmp_txtA.replace('<yearQ>',tmp_lt[0])
			tmp_txtA = tmp_txtA.replace('<disease_Name>',tmp_lt[1])
			tmp_txtA = tmp_txtA.replace('<disease_CUI>',tmp_lt[2])
			tmp_txtA = tmp_txtA.replace('<related_CUI>',tmp_lt[3])
			tmp_txtA = tmp_txtA.replace('<<terms/expressions>>',tmp_lt[4])
			tmp_txtA = tmp_txtA.replace('<paragraph>',tmp_lt[5])
			tmp_txtA = tmp_txtA.replace('<pub_Title>',tmp_lt[6])
			tmp_txtA = tmp_txtA.replace('<pub_DateIssued>',tmp_lt[7])
			tmp_txtA = tmp_txtA.replace('<pub_Identifier>',tmp_lt[8])

			if (tmp_cntTOT == 1):
				tmp_cnt = len(tmp_l)
		#---
		tmp_cnt = tmp_cnt + 1
	#--- --- --- --- ---
	###---print(str(tmp_cntTOT))
	print("all done! have a nice day! :-))")

	tmp_f1=open(os.path.expanduser(fno), "w")
	tmp_f1.write("\n" + tmp_txtA + "\n")
	tmp_f1.close()
#-------------------------------------EndFunction

###--- --- --- excute the 1 line below [ get Only the codes from a VOC source] [ useful for creating an ontology signature (set of entities) ]
RgetMRCONSOcodesSCT(EXP_dir + 'MRCONSO_sample.RRF', ['C0034362','C0010240'], 'ENG', 'SNOMEDCT_US', EXP_dir + 'v1_codes2CUIs-Qfever__pub-zoo.txt')

###--- --- --- excute the 1 line below [ get information about CUIs ][ any source vocabularies represented in the UMLS ]
RgetMRCONSOInfo(EXP_dir + 'MRCONSO_sample.RRF', ['C0034362','C0010240'], 'ENG', '', EXP_dir + 'v2_info2CUIs-Qfever-EnglishUMLS__pub-zoo.txt')

###--- --- --- excute the 1 line below [ get information about CUIs ][ this information is needed for creating value sets ]
RgetMRCONSOInfo(EXP_dir + 'MRCONSO_sample.RRF', ['C0034362','C0010240'], 'ENG', 'SNOMEDCT_US', EXP_dir + 'v3_info2CUIs-Qfever__pub-zoo.txt')

###--- --- --- excute the 1 line below to create a FHIR ValueSet instance in JSON [ the input file was created with RgetMRCONSOInfo() ]
RcreateFHIRvs('JSON', EXP_dir + 'codes-full_Qfever_pub-zoo.txt', 'vsQfever')

###--- --- --- excute the 1 line below to create a FHIR ValueSet instance in RDF Turtle [ the input file was created with RgetMRCONSOInfo() ]
RcreateFHIRvs('RDF', EXP_dir + 'codes-full_Qfever_pub-zoo.txt', 'vsQfever')

###--- --- --- excute the 1 line below to create a FHIR ValueSet instance in non-normative JSON-LD [ the input file was created with RgetMRCONSOInfo() ]
RcreateFHIRvs('JSON-LD', EXP_dir + 'codes-full_Qfever_pub-zoo.txt', 'vsQfever')

###--- --- --- excute the 1 line below to create a nanoPublication in JSON-LD [ a dummy ORCID is used to identify the creator of the nanoPublication ]
RcreateNanoPub(EXP_dir + '5_2024Q2.txt','0000-0000-0000-0001')