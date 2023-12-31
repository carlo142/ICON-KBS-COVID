'''
'''
 
from esame.src.user import User

def interaction():
    print("Benvenuto nel sistema per far previsioni se sei affetto dal virus SARS-CoV-2\n\
    Per capire cio' ci sono una serie di domande su patologie pregresse e sintomi che potreste avere\n\
    Vi preghiamo di rispondere con la massima sincerita'\n\
    Grazie\n")
    print("Le risposte devono essere date nei possibili seguenti modi:\n\
    ->si-s-yes-y\n\
    ->no-n\n")
    utente = User()
    
    while(True):
        print('Hai problemi di respirazione?')
        utente.setBreathingProblem(input(''))
        if(utente.breathingProblem.lower().__eq__('si') or utente.breathingProblem.lower().__eq__('s') or utente.breathingProblem.lower().__eq__('y') or utente.breathingProblem.lower().__eq__('yes')):
            utente.setBreathingProblem('Yes')
            break
        elif(utente.breathingProblem.lower().__eq__('no') or utente.breathingProblem.lower().__eq__('n')):
            utente.setBreathingProblem('No')
            break
        
    while(True): 
        print("Hai febbre?")
        utente.setFever(input(''))
        if(utente.fever.lower().__eq__('si') or utente.fever.lower().__eq__('s') or utente.fever.lower().__eq__('y') or utente.fever.lower().__eq__('yes')):
            utente.setFever('Yes')
            break
        elif(utente.fever.lower().__eq__('no') or utente.fever.lower().__eq__('n')):
            utente.setFever('No')
            break
    
    while(True):
        print("Hai tosse secca?")
        utente.setDryCought(input(''))
        if(utente.dryCought.lower().__eq__('si') or utente.dryCought.lower().__eq__('s') or utente.dryCought.lower().__eq__('y') or utente.dryCought.lower().__eq__('yes')):
            utente.setDryCought('Yes')
            break
        elif(utente.dryCought.lower().__eq__('no') or utente.dryCought.lower().__eq__('n')):
            utente.setDryCought('No')
            break
        
    while(True):
        print("Hai mal di gola?")
        utente.setSoreThroat(input(''))
        if(utente.soreThroat.lower().__eq__('si') or utente.soreThroat.lower().__eq__('s') or utente.soreThroat.lower().__eq__('y') or utente.soreThroat.lower().__eq__('yes')):
            utente.setSoreThroat('Yes')
            break
        elif(utente.soreThroat.lower().__eq__('no') or utente.soreThroat.lower().__eq__('n')):
            utente.setSoreThroat('No')
            break
        
    while(True):    
        print("Hai asma?")
        utente.setAsthma(input(''))
        if(utente.asthma.lower().__eq__('si') or utente.asthma.lower().__eq__('s') or utente.asthma.lower().__eq__('y') or utente.asthma.lower().__eq__('yes')):
            utente.setAsthma('Yes')
            break
        elif(utente.asthma.lower().__eq__('no') or utente.asthma.lower().__eq__('n')):
            utente.setAsthma('No')
            break
     
    while(True):    
        print("Hai malattia del cuore?")
        utente.setHeartDisease(input(''))
        if(utente.heartDisease.lower().__eq__('si') or utente.heartDisease.lower().__eq__('s') or utente.heartDisease.lower().__eq__('y') or utente.heartDisease.lower().__eq__('yes')):
            utente.setHeartDisease('Yes')
            break
        elif(utente.heartDisease.lower().__eq__('no') or utente.heartDisease.lower().__eq__('n')):
            utente.setHeartDisease('No')
            break
        
    while(True):
        print("Hai diabete?")
        utente.setDiabetes(input(''))
        if(utente.diabetes.lower().__eq__('si') or utente.diabetes.lower().__eq__('s') or utente.diabetes.lower().__eq__('y') or utente.diabetes.lower().__eq__('yes')):
            utente.setDiabetes('Yes')
            break
        elif(utente.diabetes.lower().__eq__('no') or utente.diabetes.lower().__eq__('n')):
            utente.setDiabetes('No')
            break
        
    while(True):
        print("Hai ipertensione?")
        utente.setHyperTension(input(''))
        if(utente.hyperTension.lower().__eq__('si') or utente.hyperTension.lower().__eq__('s') or utente.hyperTension.lower().__eq__('y') or utente.hyperTension.lower().__eq__('yes')):
            utente.setHyperTension('Yes')
            break
        elif(utente.hyperTension.lower().__eq__('no') or utente.hyperTension.lower().__eq__('n')):
            utente.setHyperTension('No')
            break
        
    while(True):
        print("Hai viaggiato all'estero?")
        utente.setAbroadTravel(input(''))
        if(utente.abroadTravel.lower().__eq__('si') or utente.abroadTravel.lower().__eq__('s') or utente.abroadTravel.lower().__eq__('y') or utente.abroadTravel.lower().__eq__('yes')):
            utente.setAbroadTravel('Yes')
            break
        elif(utente.abroadTravel.lower().__eq__('no') or utente.abroadTravel.lower().__eq__('n')):
            utente.setAbroadTravel('No')
            break
         
    while(True):
        print("Hai avuto contatti con pazienti COVID?")
        utente.setContactCovidPatient(input(''))
        if(utente.contactCovidPatient.lower().__eq__('si') or utente.contactCovidPatient.lower().__eq__('s') or utente.contactCovidPatient.lower().__eq__('y') or utente.contactCovidPatient.lower().__eq__('yes')):
            utente.setContactCovidPatient('Yes')
            break
        elif(utente.contactCovidPatient.lower().__eq__('no') or utente.contactCovidPatient.lower().__eq__('n')):
            utente.setContactCovidPatient('No')
            break
        
    while(True):
        print("Hai partecipato ad un grande raduno?")
        utente.setAttendedLargeGathering(input(''))
        if(utente.attendedLargeGathering.lower().__eq__('si') or utente.attendedLargeGathering.lower().__eq__('s') or utente.attendedLargeGathering.lower().__eq__('y') or utente.attendedLargeGathering.lower().__eq__('yes')):
            utente.setAttendedLargeGathering('Yes')
            break
        elif(utente.attendedLargeGathering.lower().__eq__('no') or utente.attendedLargeGathering.lower().__eq__('n')):
            utente.setAttendedLargeGathering('No')
            break
        
    while(True):
        print("Hai visitato posti esposti al pubblico?")
        utente.setVisitedPublicPlaces(input(''))
        if(utente.visitedPublicPlaces.lower().__eq__('si') or utente.visitedPublicPlaces.lower().__eq__('s') or utente.visitedPublicPlaces.lower().__eq__('y') or utente.visitedPublicPlaces.lower().__eq__('yes')):
            utente.setVisitedPublicPlaces('Yes')
            break
        elif(utente.visitedPublicPlaces.lower().__eq__('no') or utente.visitedPublicPlaces.lower().__eq__('n')):
            utente.setVisitedPublicPlaces('No')
            break
        
    while(True):
        print("La tua famiglia lavora in luoghi esposti al pubblico?")
        utente.setFamilyPublicPlaces(input(''))
        if(utente.familyPublicPlaces.lower().__eq__('si') or utente.familyPublicPlaces.lower().__eq__('s') or utente.familyPublicPlaces.lower().__eq__('y') or utente.familyPublicPlaces.lower().__eq__('yes')):
            utente.setFamilyPublicPlaces('Yes')
            break
        elif(utente.familyPublicPlaces.lower().__eq__('no') or utente.familyPublicPlaces.lower().__eq__('n')):
            utente.setFamilyPublicPlaces('No')
            break
  
    return utente