  
% 
%Purple = New 


%import data, clean up variables
 
    format long g %gets rid of scientific notation hell yeah!!!!
    load('allUSD_clean.mat');
    allUSD = sortrows(allUSD,'ID','descend'); %using ID to sort in order to randomize data
    	numSamples = size(allUSD,1);


    %VARIABLES VECTORS

        betterIndex = [1:size(allUSD,1)]'; 
        backers = table2array(allUSD(:,11));
        pledged = table2array(allUSD(:,14));
        goal = table2array(allUSD(:,15));
        loggoal = log(goal);
        loggoalsquared = loggoal.^2;


        
%Calculate duration
        launched = allUSD(:,8);
        launched = table2array(launched);
        deadline = table2array(allUSD(:,6));
        launched = datetime(launched,'InputFormat', 'MM/dd/yyyy HH:mm');
        deadline = datetime(deadline,'InputFormat','MM/dd/yyyy');
        duration = deadline-launched;
        duration = days(duration);
        
        rateOfBackers = backers./duration;
        amountPerBacker = pledged./backers;
        percentFunded = pledged./goal;

        %Interactions
        duraback = duration.*backers;
        goaback = goal.*backers;
        goadura = goal.*duration; 


        %main categories:
            art = allUSD.main_category == 'Art';
            comics = allUSD.main_category == 'Comics';
            crafts = allUSD.main_category == 'Crafts';
            dance = allUSD.main_category == 'Dance';
            design = allUSD.main_category == 'Design';
            fashion = allUSD.main_category == 'Fashion';
            film = allUSD.main_category == 'Film & Video';
            food = allUSD.main_category == 'Food';
            games = allUSD.main_category == 'Games';
            journalism = allUSD.main_category == 'Journalism';
            music = allUSD.main_category == 'Music';
            photography = allUSD.main_category == 'Photography';
            publishing = allUSD.main_category == 'Publishing';
            tech = allUSD.main_category == 'Technology';
            theater = allUSD.main_category == 'Theater';
            categoriesLogical = [art comics crafts dance design fashion film food games journalism music photography publishing tech theater];
        
    %recreate matrix with variables of interest as arrays (instead of
 	BIGGERBOI = [percentFunded backers pledged loggoal duration rateOfBackers amountPerBacker categoriesLogical loggoalsquared duraback goaback goadura];
        BIGGERBOI(isnan(BIGBOI))=0;
        %remove outliers with PF>100.00
        BIGBOI = BIGBOI(BIGBOI(:,1)<10,:);

 
        xVals = ["Percent Funded", "Backers", "Pledged", "Goal", "Duration", "Rate of Backers", "Amount Per Backer", "Art", "Comics", "Crafts", "Dance", "Design", "Fashion", "Film & Video", "Food", "Games", "Journalism", "Music", "Photography", "Publishing", "Technology", "Theater", "Goal Squared", "DurationXBackers", "GoalXBackers", "GoalXDurations"];

        
%FUN. WITH. GRAPHS. 
 
 
%sort into training set and test set
    allTraining = BIGBOI(1:round(.8*numSamples),:);
        training = allTraining(:,2:26);
        pftrain = allTraining(:,1); %percent funded for training 
    allTest = BIGBOI(size(training,1)+1:numSamples,:);
        test = allTest(:,2:26);
        pftest = allTest(:,1); %percent funded for test

    
%creating a ones vector 
    onesVector = ones(size(training,1),1);

% SORTING THE DATA BASED ON CATEGORIES 

%%
%Sort data for ART  
target = 1; 
art_temp = find(BIGBOI(:,8) == target);
art_sort = (BIGBOI(art_temp,1:7));
clear art_temp
 
%%
%sort data for comics
 
comics_temp = find(BIGBOI(:,9) == target);
comics_sort = (BIGBOI(comics_temp,1:7));
clear comics_temp 
%%
%sort Craft data 
craft_temp = find(BIGBOI(:,10) == target); 
craft_sort = (BIGBOI(craft_temp,1:7));
clear craft_temp 
%%
%sort dance data 
dance_temp = find(BIGBOI(:,11) == target); 
dance_sort = (BIGBOI(dance_temp,1:7));
clear dance_temp 
 
%%
%Sort for design 
design_temp = find(BIGBOI(:,12) == target); 
design_sort = (BIGBOI(design_temp,1:7));
clear design_temp
 
%% 
%Sort for fashion 
fashion_temp = find(BIGBOI(:,13) == target); 
fashion_sort = (BIGBOI(fashion_temp,1:7));
clear craft_temp 
 
%%
%film 
film_temp = find(BIGBOI(:,14) == target); 
film_sort = (BIGBOI(film_temp,1:7));
clear film_temp 
 
%%
% food
food_temp = find(BIGBOI(:,15) == target); 
food_sort = (BIGBOI(food_temp,1:7));
clear food_temp 
 
%%
%games 
 
games_temp = find(BIGBOI(:,16) == target); 
games_sort = (BIGBOI(games_temp,1:7));
clear games_temp
 
%%
%journalism 
 
journalism_temp = find(BIGBOI(:,17) == target); 
journalism_sort = (BIGBOI(journalism_temp,1:7));
clear journalism_temp
 
%%
% music 
 
music_temp = find(BIGBOI(:,18) == target); 
music_sort = (BIGBOI(music_temp,1:7));
clear music_temp
 
%%
%photography 
 
photo_temp = find(BIGBOI(:,19) == target); 
photo_sort = (BIGBOI(photo_temp,1:7));
clear photo_temp
 
%%
%publishing 
publishing_temp = find(BIGBOI(:,20) == target); 
publishing_sort = (BIGBOI(publishing_temp,1:7));
clear publishing_temp
 
%%
%tech 
 
tech_temp = find(BIGBOI(:,21) == target); 
tech_sort = (BIGBOI(tech_temp,1:7));
clear tech_temp
 
%%
%theater
theater_temp = find(BIGBOI(:,22) == target); 
theater_sort = (BIGBOI(theater_temp,1:7));
clear theater_temp

%% 
%HERE THAR BE MODELS
 
    %M1: Backers, duration, rateofbackers
        A1 = [onesVector training(:,1) training(:,4) training(:,5)];
        w1 = A1\pftrain; 
    %M2: backers, duration, categories (DOESN'T WORK)
        A2noise = [zeros(231870,3) normrnd(0,0.01,231870,15)];
        A2 = [onesVector training(:,1) training(:,4) training(:,7:21)];
        A2noise = A2noise + A2;
        w2 = A2noise\pftrain;
        %%
    %M3: backers, goal, duration, categories
    %(DOESN'T WORK)
        A3noise = [zeros(231870,4) normrnd(0,0.01,231870,15)];
        A3 = [onesVector training(:,1) training(:,3) training(:,4) training(:,7:21)];
        A3noise = A3noise + A3;
        w3 = A3noise\pftrain;
        %%
    %M4: backers, goal, goalsquared, duration, categories
        A4noise = [zeros(231870,5) normrnd(0,0.01,231870,15)];
        A4 = [onesVector training(:,1) training(:,3) training(:, 22), training(:,4) training(:,7:21)];
        A4noise = A4noise + A4;
        w4 = A4noise\pftrain;
        
        %%
    %M5: backers, goal, goalsquared, goaback, duration, duraback, goaback, goadura, categories, 
        A5noise = [zeros(231870,8) normrnd(0,0.02,231870,15)];
        A5 = [onesVector training(:,1) training(:,3) training(:,22) training(:,4) training(:,23:25) training(:,7:21) ];
        A5noise = A5noise + A5;
        w5 = A5noise\pftrain;
    
%calculate SSE for training data
    yhat1 = A1*w1;
    
    yhat2 = A2*w2;
    yhat3 = A3*w3;
    yhat4 = A4*w4;
    yhat5 = A5*w5;
    sse1 = sum((yhat1-pftrain).^2);
    sse2 = sum((yhat2-pftrain).^2);
    sse3 = sum((yhat3-pftrain).^2);
    sse4 = sum((yhat4-pftrain).^2);
    sse5 = sum((yhat5-pftrain).^2);
    
%histogram of sse for training data
    ssearray = [sse1 sse2 sse3 sse4 sse5];
    figure
    bar([1 2 3 4 5], ssearray)
%%
%run test data through Models
    testOnesVector = ones(size(test,1),1);
  %M1  
    A1Test = [testOnesVector test(:,1) test(:,4) test(:,5)];
  %M2
    A2Testnoise = [zeros(size(test,1),3) normrnd(0,0.01,size(test,1),15)];
    A2Test = [testOnesVector test(:,1) test(:,4) test(:,7:21)];
    A2Testnoise = A2Testnoise + A2Test;
  %M3
    A3Testnoise = [zeros(size(test,1),4) normrnd(0,0.01,size(test,1),15)];
    A3Test = [testOnesVector test(:,1) test(:,3) test(:,4) test(:,7:21)];
    A3Testnoise = A3Testnoise + A3Test;
  %M4
    A4Testnoise = [zeros(size(test,1),5) normrnd(0,0.01,size(test,1),15)];
    A4Test = [testOnesVector test(:,1) test(:,3) test(:, 22), test(:,4) test(:,7:21)];
    A4Testnoise = A4Testnoise + A4Test;
  %M5
    A5Testnoise = [zeros(size(test,1),8) normrnd(0,0.02,size(test,1),15)];
    A5Test = [testOnesVector test(:,1) test(:,3) test(:,22) test(:,4) test(:,23:25) test(:,7:21) ];
    A5Testnoise = A5Testnoise + A5Test;
    
%calculate SSE for test data    
    yhat1Test = A1Test*w1;
    yhat2Test = A2Testnoise*w2;
    yhat3Test = A3Testnoise*w3;
    yhat4Test = A4Testnoise*w4;
    yhat5Test = A5Testnoise*w4;
    sse1test = sum((yhat1Test-pftest).^2);
    sse2test = sum((yhat2Test-pftest).^2);
    sse3test = sum((yhat3Test-pftest).^2);
    sse4test = sum((yhat4Test-pftest).^2);
    sse5test = sum((yhat5Test-pftest).^2);
    
%histograms of test data
    sseTestarray = [sse1test sse2test sse3test sse4test sse5test];
    figure
    bar([1 2 3 4 5], sseTestarray)
%sseCompare = [ssearray; sseTestarray]'
    %figure
    %bar(sseCompare)



%%FOR PLOTTING
   
     



SYDNEY?S FINALISH VERSION

 
 
%import data, clean up variables
 
    format long g %gets rid of scientific notation hell yeah!!!!
    load('allUSD_clean.mat');
    allUSD = sortrows(allUSD,'ID','descend'); %using ID to sort in order to randomize data
 
    %VARIABLES VECTORS
        betterIndex = [1:size(allUSD,1)]'; 
        backers = table2array(allUSD(:,11));
        pledged = table2array(allUSD(:,14));
        goal = table2array(allUSD(:,15));
        loggoal = log(goal);
        loggoalsquared = loggoal.^2;
        
%Calculate duration
 
        launched = allUSD(:,8);
        launched = table2array(launched);
        deadline = table2array(allUSD(:,6));
        launched = datetime(launched,'InputFormat', 'MM/dd/yyyy HH:mm');
        deadline = datetime(deadline,'InputFormat','MM/dd/yyyy');
        duration = deadline-launched;
        duration = days(duration);
        
        rateOfBackers = backers./duration;
        amountPerBacker = pledged./backers;
        percentFunded = pledged./goal;
        
        %Interactions
        duraback = duration.*backers;
        goaback = goal.*backers;
        goadura = goal.*duration;     
 
        %main categories:
            art = allUSD.main_category == 'Art';
            comics = allUSD.main_category == 'Comics';
            crafts = allUSD.main_category == 'Crafts';
            dance = allUSD.main_category == 'Dance';
            design = allUSD.main_category == 'Design';
            fashion = allUSD.main_category == 'Fashion';
            film = allUSD.main_category == 'Film & Video';
            food = allUSD.main_category == 'Food';
            games = allUSD.main_category == 'Games';
            journalism = allUSD.main_category == 'Journalism';
            music = allUSD.main_category == 'Music';
            photography = allUSD.main_category == 'Photography';
            publishing = allUSD.main_category == 'Publishing';
            tech = allUSD.main_category == 'Technology';
            theater = allUSD.main_category == 'Theater';
            categoriesLogical = [art comics crafts dance design fashion film food games journalism music photography publishing tech theater];
   
    %recreate matrix with variables of interest as arrays (instead of
    %tables 
        BIGGERBOI = [percentFunded backers pledged loggoal duration rateOfBackers amountPerBacker categoriesLogical loggoalsquared duraback goaback goadura];
        BIGGERBOI(isnan(BIGGERBOI))=0;%remove outliers with PF>100.00
        BIGBOI = BIGGERBOI(BIGGERBOI(:,1)<10,:);
        BIGBOI(isnan(BIGBOI))=0;
        numSamples = size(BIGBOI,1);
        xVals = ["Percent Funded", "Backers", "Pledged", "Goal", "Duration", "Rate of Backers", "Amount Per Backer", "Art", "Comics", "Crafts", "Dance", "Design", "Fashion", "Film & Video", "Food", "Games", "Journalism", "Music", "Photography", "Publishing", "Technology", "Theater", "Goal Squared", "DurationXBackers", "GoalXBackers", "GoalXDurations"];
 
%%
 
%sort into training set and test set
    allTraining = BIGBOI(1:round(.8*numSamples),:);
        training = allTraining(:,2:26);
        pftrain = allTraining(:,1); %percent funded for training 
    allTest = BIGBOI(size(training,1)+1:numSamples,:);
        test = allTest(:,2:26);
        pftest = allTest(:,1); %percent funded for test
       
%creating a ones vector 
    onesVector = ones(size(training,1),1); 
 
%HERE THAR BE MODELS
 
    %M1: Backers, duration, rateofbackers
        A1 = [onesVector training(:,1) training(:,4) training(:,5)];
        w1 = A1\pftrain; 
    %M2: backers, duration, categories (DOESN'T WORK)
        A2noise = [zeros(size(training,1),3) normrnd(0,0.01,size(training,1),15)];
        A2 = [onesVector training(:,1) training(:,4) training(:,7:21)];
        A2noise = A2noise + A2;
        w2 = A2noise\pftrain;
        %%
    %M3: backers, goal, duration, categories
    %(DOESN'T WORK)
        A3noise = [zeros(size(training,1),4) normrnd(0,0.01,size(training,1),15)];
        A3 = [onesVector training(:,1) training(:,3) training(:,4) training(:,7:21)];
        A3noise = A3noise + A3;
        w3 = A3noise\pftrain;
        %%
    %M4: backers, goal, goalsquared, duration, categories
        A4noise = [zeros(size(training,1),5) normrnd(0,0.01,size(training,1),15)];
        A4 = [onesVector training(:,1) training(:,3) training(:, 22), training(:,4) training(:,7:21)];
        A4noise = A4noise + A4;
        w4 = A4noise\pftrain;
        
        %%
    %M5: backers, goal, goalsquared, goaback,  duration, duraback, goaback, goadura, categories, 
        A5noise = [zeros(size(training,1),8) normrnd(0,0.02,size(training,1),15)];
        A5 = [onesVector training(:,1) training(:,3) training(:,22) training(:,4) training(:,23:25) training(:,7:21) ];
        A5noise = A5noise + A5;
        w5 = A5noise\pftrain;
    
%calculate SSE for training data
    yhat1 = A1*w1;
    yhat2 = A2*w2;
    yhat3 = A3*w3;
    yhat4 = A4*w4;
    yhat5 = A5*w5;
    sse1 = sum((yhat1-pftrain).^2);
    sse2 = sum((yhat2-pftrain).^2);
    sse3 = sum((yhat3-pftrain).^2);
    sse4 = sum((yhat4-pftrain).^2);
    sse5 = sum((yhat5-pftrain).^2);
    
%%
%run test data through Models
    testOnesVector = ones(size(test,1),1);
  %M1  
    A1Test = [testOnesVector test(:,1) test(:,4) test(:,5)];
  %M2
    A2Testnoise = [zeros(size(test,1),3) normrnd(0,0.01,size(test,1),15)];
    A2Test = [testOnesVector test(:,1) test(:,4) test(:,7:21)];
    A2Testnoise = A2Testnoise + A2Test;
  %M3
    A3Testnoise = [zeros(size(test,1),4) normrnd(0,0.01,size(test,1),15)];
    A3Test = [testOnesVector test(:,1) test(:,3) test(:,4) test(:,7:21)];
    A3Testnoise = A3Testnoise + A3Test;
  %M4
    A4Testnoise = [zeros(size(test,1),5) normrnd(0,0.01,size(test,1),15)];
    A4Test = [testOnesVector test(:,1) test(:,3) test(:, 22), test(:,4) test(:,7:21)];
    A4Testnoise = A4Testnoise + A4Test;
  %M5
    A5Testnoise = [zeros(size(test,1),8) normrnd(0,0.02,size(test,1),15)];
    A5Test = [testOnesVector test(:,1) test(:,3) test(:,22) test(:,4) test(:,23:25) test(:,7:21) ];
    A5Testnoise = A5Testnoise + A5Test;
    
%calculate SSE for test data    
    yhat1Test = A1Test*w1;
    yhat2Test = A2Testnoise*w2;
    yhat3Test = A3Testnoise*w3;
    yhat4Test = A4Testnoise*w4;
    yhat5Test = A5Testnoise*w5;
    sse1test = sum((yhat1Test-pftest).^2);
    sse2test = sum((yhat2Test-pftest).^2);
    sse3test = sum((yhat3Test-pftest).^2);
    sse4test = sum((yhat4Test-pftest).^2);
    sse5test = sum((yhat5Test-pftest).^2);
  
%FUN. WITH. GRAPHS. 
% SORTING THE DATA BASED ON CATAGORIES 
 
%%
%Sort data for ART  
target = 1; 
art_temp = find(BIGBOI(:,8) == target);
art_sort = (BIGBOI(art_temp,1:7));
art_success = size(art_sort(art_sort(:,1)>=1,:),1);
art_fail = size(art_sort(art_sort(:,1)<1,:),1);
clear art_temp
 
%%
%sort data for comics
 
comics_temp = find(BIGBOI(:,9) == target);
comics_sort = (BIGBOI(comics_temp,1:7));
comics_success = size(comics_sort(comics_sort(:,1)>=1,:),1);
comics_fail = size(comics_sort(comics_sort(:,1)<1,:),1);
clear comics_temp 
%%
%sort Craft data 
craft_temp = find(BIGBOI(:,10) == target); 
craft_sort = (BIGBOI(craft_temp,1:7));
craft_success = size(craft_sort(craft_sort(:,1)>=1,:),1);
craft_fail = size(craft_sort(craft_sort(:,1)<1,:),1);
clear craft_temp 
%%
%sort dance data 
dance_temp = find(BIGBOI(:,11) == target); 
dance_sort = (BIGBOI(dance_temp,1:7));
dance_success = size(dance_sort(dance_sort(:,1)>=1,:),1);
dance_fail = size(dance_sort(dance_sort(:,1)<1,:),1);
clear dance_temp 
 
%%
%Sort for design 
design_temp = find(BIGBOI(:,12) == target); 
design_sort = (BIGBOI(design_temp,1:7));
design_success = size(design_sort(design_sort(:,1)>=1,:),1);
design_fail = size(design_sort(design_sort(:,1)<1,:),1);
clear design_temp
 
%% 
%Sort for fashion 
fashion_temp = find(BIGBOI(:,13) == target); 
fashion_sort = (BIGBOI(fashion_temp,1:7));
fashion_success = size(fashion_sort(fashion_sort(:,1)>=1,:),1);
fashion_fail = size(fashion_sort(fashion_sort(:,1)<1,:),1);
clear craft_temp 
 
%%
%film 
film_temp = find(BIGBOI(:,14) == target); 
film_sort = (BIGBOI(film_temp,1:7));
film_success = size(film_sort(film_sort(:,1)>=1,:),1);
film_fail = size(film_sort(film_sort(:,1)<1,:),1);
clear film_temp 
 
%%
% food
food_temp = find(BIGBOI(:,15) == target); 
food_sort = (BIGBOI(food_temp,1:7));
food_success = size(food_sort(food_sort(:,1)>=1,:),1);
food_fail = size(food_sort(food_sort(:,1)<1,:),1);
clear food_temp 
 
%%
%games 
 
games_temp = find(BIGBOI(:,16) == target); 
games_sort = (BIGBOI(games_temp,1:7));
games_success = size(games_sort(games_sort(:,1)>=1,:),1);
games_fail = size(games_sort(games_sort(:,1)<1,:),1);
clear games_temp
 
%%
%journalism 
 
journalism_temp = find(BIGBOI(:,17) == target); 
journalism_sort = (BIGBOI(journalism_temp,1:7));
journalism_success = size(journalism_sort(journalism_sort(:,1)>=1,:),1);
journalism_fail = size(journalism_sort(journalism_sort(:,1)<1,:),1);
clear journalism_temp
 
%%
% music 
 
music_temp = find(BIGBOI(:,18) == target); 
music_sort = (BIGBOI(music_temp,1:7));
music_success = size(music_sort(music_sort(:,1)>=1,:),1);
music_fail = size(music_sort(music_sort(:,1)<1,:),1);
clear music_temp
 
%%
%photography 
 
photo_temp = find(BIGBOI(:,19) == target); 
photo_sort = (BIGBOI(photo_temp,1:7));
photo_success = size(photo_sort(photo_sort(:,1)>=1,:),1);
photo_fail = size(photo_sort(photo_sort(:,1)<1,:),1);
clear photo_temp
 
%%
%publishing 
publishing_temp = find(BIGBOI(:,20) == target); 
publishing_sort = (BIGBOI(publishing_temp,1:7));
publishing_success = size(publishing_sort(publishing_sort(:,1)>=1,:),1);
publishing_fail = size(publishing_sort(publishing_sort(:,1)<1,:),1);
clear publishing_temp
 
%%
%tech 
 
tech_temp = find(BIGBOI(:,21) == target); 
tech_sort = (BIGBOI(tech_temp,1:7));
tech_success = size(tech_sort(tech_sort(:,1)>=1,:),1);
tech_fail = size(tech_sort(tech_sort(:,1)<1,:),1);
clear tech_temp
 
%%
%theater
theater_temp = find(BIGBOI(:,22) == target); 
theater_sort = (BIGBOI(theater_temp,1:7));
theater_success = size(theater_sort(theater_sort(:,1)>=1,:),1);
theater_fail = size(theater_sort(theater_sort(:,1)<1,:),1);
clear theater_temp
%%
 
categoryNames = {'Art'; 'Comics'; 'Crafts'; 'Dance'; 'Design'; 'Fashion'; 'Film & Video'; 'Food'; 'Games'; 'Journalism'; 'Music'; 'Photography'; 'Publishing'; 'Technology'; 'Theater'};
 
KSblue = round([03 71 82]./256,2);
 
%bargraph of sses
figure
 
    %for training data
    subplot(2,1,1)
    ssearray = [sse1 sse2 sse3 sse4 sse5];
    b1 = bar([1 2 3 4 5], ssearray);
    b1.FaceColor = KSblue;
    title("Training Data SSEs")
    xlabel("Model #")
    
    %for test data
    subplot(2,1,2)
    sseTestarray = [sse1test sse2test sse3test sse4test sse5test];
    b2 = bar([1 2 3 4 5], sseTestarray,'FaceColor',KSblue)
    b2.FaceColor = KSblue;
    title("Test Data SSEs")
    xlabel("Model #")
%%   
%bar chart of categories for success vs fail
    successes = [art_success, comics_success, craft_success, dance_success, design_success, fashion_success, film_success, food_success, games_success, journalism_success, music_success, photo_success, publishing_success,tech_success, theater_success];
    failures = [art_fail, comics_fail, craft_fail, dance_fail, design_fail, fashion_fail, film_fail, food_fail, games_fail, journalism_fail, music_fail, photo_fail, publishing_fail,tech_fail, theater_fail];
    figure
    successCompare = [successes; failures]';
    b = bar(successCompare)
    names = gca;
    names.XTickLabel = categoryNames
    b(1).FaceColor = 'g'
    b(2).FaceColor = 'r'
    title("Successes Vs. Failures by Category")
    ylabel("Frequency")
    legend("Success", "Failure")
 %%   
%histogram of normalized goal
    figure
    h1 = histogram(loggoal);
    title("Histogram of Normalized Goal Amounts")
    xlabel("log(Goal)")
    h1.FaceColor = KSblue;
    
%histogram of percent funded (under 1000%)
    figure 
    PFmini = BIGBOI(BIGBOI(:,1)<3);
    h2 = histogram(PFmini);
    title("Histogram of Percent Funded (under 1000%)")
    xlabel("Percent Funded")
    h2.FaceColor = KSblue;



