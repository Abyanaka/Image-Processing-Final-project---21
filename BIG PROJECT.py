import cv2
import numpy as np
from keras.models import load_model

model = load_model("KARTU.h5")
cap = cv2.VideoCapture(1)

cards_per_set = 2
p1_cards = []
pC_cards = []
card_counter = 0
player_turn = True  
gameSet = False
roundFin = False
round2fin = False

while True:
    ret, frame = cap.read()

    if not ret:
        break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    blur = cv2.GaussianBlur(hsv, (5, 5), 0)
    tepiDet = cv2.Canny(blur, 50, 150)

    _, thresh = cv2.threshold(tepiDet, 240, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    labels = ("1C", "2C", "3C", "4C", "5C", "6C", "7C", "8C", "9C", "10C", "KC", "JC", "QC",
              "1S", "2S", "3S", "4S", "5S", "6S", "7S", "8S", "9S", "10S", "KS", "JS", "QS",
              "AD", "2D", "3D", "4D", "5D", "6D", "7D", "8D", "9D", "10D", "KD", "JD", "QD",
              "AH", "2H", "3H", "4H", "5H", "6H", "7H", "8H", "9H", "10H", "KH", "JH", "QH", "Closed")
    
    labels_val ={"1C" : 1, "2C" : 2, "3C" : 3, "4C" : 4, "5C" : 5, "6C" : 6, "7C" : 7, "8C" : 8, "9C" : 9, "10C" : 10, "KC" : 10, "JC" : 10, "QC" : 10,
                 "1S" : 1, "2S" : 2, "3S" : 3, "4S" : 4, "5S" : 5, "6S" : 6, "7S" : 7, "8S" : 8, "9S" : 9, "10S" : 10, "KS" : 10, "JS" : 10, "QS" : 10,
                 "AD" : 1, "2D" : 2, "3D" : 3, "4D" : 4, "5D" : 5, "6D" : 6, "7D" : 7, "8D" : 8, "9D" : 9, "10D" : 10, "KD" : 10, "JD" : 10, "QD" : 10,
                 "AH" : 1, "2H" : 2, "3H" : 3, "4H" : 4, "5H" : 5, "6H" : 6, "7H" : 7, "8H" : 8, "9H" : 9, "10H" : 10, "KH" : 10, "JH" : 10, "QH" : 10,
                 "Closed" : 0}

    for i in range(cards_per_set):
            if i < len(p1_cards):
                label = p1_cards[i]["label"]
                text = f"P1: {label}"
                cv2.putText(frame, text, (0, 100 + i * 30), cv2.FONT_HERSHEY_TRIPLEX, 0.75, (255, 255, 255), 2)

    for contour in contours:
        lengkungan = 0.04 * cv2.arcLength(contour, True)
        tepiKartu = cv2.approxPolyDP(contour, lengkungan, True)

        if len(tepiKartu) == 4 and cv2.contourArea(contour) > 24000:
            x, y, w, h = cv2.boundingRect(tepiKartu)
            detectedCard = frame[y:y + h, x:x + w]
            resizeCard = cv2.resize(detectedCard, (128, 128))

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(frame, "card", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            resizeCard = resizeCard.astype('float32') / 255
            resizeCard = np.expand_dims(resizeCard, axis=0)

            prediction = model.predict(resizeCard)
            max_index = np.argmax(prediction)
            closest_label = labels[max_index]

            if player_turn:
                if closest_label not in [card["label"] for card in p1_cards] and closest_label not in [card["label"] for card in pC_cards]:
                    cv2.waitKey(2000)
                    p1_cards.append({"label": closest_label})
                    card_counter += 1
                elif closest_label == "Closed":
                    cv2.waitKey(2000)
                    p1_cards.append({"label": closest_label})
                    card_counter += 1

                if card_counter == cards_per_set:
                    key = cv2.waitKey(0)
                    if key == ord('d'):
                        player_turn = False
                        card_counter= 0

            else:
                if round2fin == True:
                    card_counter = cards_per_set - 1
                    if closest_label not in [card["label"] for card in pC_cards] and closest_label not in [card["label"] for card in p1_cards]:
                        cv2.waitKey(2000)
                        pC_cards.append({"label": closest_label})
                        card_counter += 1
                    elif closest_label == "Closed":
                        cv2.waitKey(2000)
                        pC_cards.append({"label": closest_label})
                        card_counter += 1

                    if card_counter == cards_per_set:
                        roundFin = True
                        round2fin = False
                else :
                    if closest_label not in [card["label"] for card in pC_cards] and closest_label not in [card["label"] for card in p1_cards]:
                        cv2.waitKey(2000)
                        pC_cards.append({"label": closest_label})
                        card_counter += 1
                    elif closest_label == "Closed":
                        cv2.waitKey(2000)
                        pC_cards.append({"label": closest_label})
                        card_counter += 1

                    if card_counter == cards_per_set:
                        roundFin = True

            cv2.putText(frame, closest_label, (x + 50, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    p1_score = sum([labels_val[card["label"]] for card in p1_cards])
    pC_score = sum([labels_val[card["label"]] for card in pC_cards])

    scoreP1Text = f"Your Score: {p1_score}"
    scorePCText = f"COM Score: Hidden"

    cv2.putText(frame, scoreP1Text, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)
    text_size = cv2.getTextSize(scoreP1Text, cv2.FONT_HERSHEY_DUPLEX, 0.75, 2)[0]
    rect_position = (8, 27 - text_size[1])
    rect_size = (text_size[0] + 30, text_size[1] + 10)
    cv2.rectangle(frame, rect_position, (rect_position[0] + rect_size[0], rect_position[1] + rect_size[1]), (200, 200, 200), 2)
    

    cv2.putText(frame, scorePCText, (400, 30), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 0), 2)
    text_size = cv2.getTextSize(scorePCText, cv2.FONT_HERSHEY_DUPLEX, 0.75, 2)[0]
    rect_position = (395, 27 - text_size[1])
    rect_size = (text_size[0] + 30, text_size[1] + 10)
    cv2.rectangle(frame, rect_position, (rect_position[0] + rect_size[0], rect_position[1] + rect_size[1]), (0, 0, 0), 2)

    if player_turn == False and roundFin == True:
        askingPosition = (75, 250)
        cv2.putText(frame, "DO YOU WANT MORE CARDS? (y/n)", askingPosition, cv2.FONT_HERSHEY_TRIPLEX, 0.75, (0, 255, 255), 2)
        PlayerText = " "
        otherPlayer = " "
        if p1_score <= 19 and pC_score <= 17:
            key = cv2.waitKey(1)
            if key == ord('y'):
                PlayerText = "Both Players : MORE CARDS PLEASE!"
                PlayerText_Posisiton = (75,350)
    
                text_size = cv2.getTextSize(PlayerText, cv2.FONT_HERSHEY_DUPLEX, 0.75, 2)[0]
                rect_position = (PlayerText_Posisiton[0], PlayerText_Posisiton[1] - text_size[1])
                rect_size = (text_size[0] + 30, text_size[1] + 10)
                cv2.rectangle(frame, rect_position, (rect_position[0] + rect_size[0], rect_position[1] + rect_size[1]), (255, 255, 255), -1)
    
                cv2.putText(frame, PlayerText, PlayerText_Posisiton, cv2.FONT_HERSHEY_TRIPLEX, 0.75, (200, 200, 200), 2)
                cv2.imshow('21', frame)
                cv2.waitKey(5000)
    
                gameSet = False
                player_turn = True
                cards_per_set += 1
                roundFin = False
                round2fin = True

            elif key == ord('n'):
                PlayerText = "COM : MORE CARDS PLEASE!"
                otherPlayer = "PLAYER 1 : I'm gonna pass"
                PlayerText_Posisiton = (75,350)
                otherPlayer_Text_Posisiton = (75, 400)

                text_size = cv2.getTextSize(PlayerText, cv2.FONT_HERSHEY_DUPLEX, 0.75, 2)[0]
                rect_position = (PlayerText_Posisiton[0], PlayerText_Posisiton[1] - text_size[1])
                rect_size = (text_size[0] + 30, text_size[1] + 10)
                cv2.rectangle(frame, rect_position, (rect_position[0] + rect_size[0], rect_position[1] + rect_size[1]), (0, 0, 0), -1)

                text_size = cv2.getTextSize(otherPlayer, cv2.FONT_HERSHEY_DUPLEX, 0.75, 2)[0]
                rect_position = (otherPlayer_Text_Posisiton[0], otherPlayer_Text_Posisiton[1] - text_size[1])
                rect_size = (text_size[0] + 30, text_size[1] + 10)
                cv2.rectangle(frame, rect_position, (rect_position[0] + rect_size[0], rect_position[1] + rect_size[1]), (255, 255, 255), -1)


                cv2.putText(frame, PlayerText, PlayerText_Posisiton, cv2.FONT_HERSHEY_TRIPLEX, 0.75, (255, 255, 255), 2)
                cv2.putText(frame, otherPlayer, otherPlayer_Text_Posisiton, cv2.FONT_HERSHEY_TRIPLEX, 0.75, (0,0,0), 2)
                cv2.imshow('21', frame)
                cv2.waitKey(5000) 

                gameSet = False
                player_turn = True
                cards_per_set +=1
                roundFin = False
                round2fin = True
            
        elif p1_score >= 20 and p1_score <= 21 and pC_score <=17:
            key = cv2.waitKey(1)
            if key == ord('n'):
                PlayerText = "COM : MORE CARDS PLEASE!"
                otherPlayer = "PLAYER 1 : I'm gonna pass"
                PlayerText_Posisiton = (75,350)
                otherPlayer_Text_Posisiton = (75, 400)

                text_size = cv2.getTextSize(PlayerText, cv2.FONT_HERSHEY_DUPLEX, 0.75, 2)[0]
                rect_position = (PlayerText_Posisiton[0], PlayerText_Posisiton[1] - text_size[1])
                rect_size = (text_size[0] + 30, text_size[1] + 10)
                cv2.rectangle(frame, rect_position, (rect_position[0] + rect_size[0], rect_position[1] + rect_size[1]), (0, 0, 0), -1)

                text_size = cv2.getTextSize(otherPlayer, cv2.FONT_HERSHEY_DUPLEX, 0.75, 2)[0]
                rect_position = (otherPlayer_Text_Posisiton[0], otherPlayer_Text_Posisiton[1] - text_size[1])
                rect_size = (text_size[0] + 30, text_size[1] + 10)
                cv2.rectangle(frame, rect_position, (rect_position[0] + rect_size[0], rect_position[1] + rect_size[1]), (255, 255, 255), -1)


                cv2.putText(frame, PlayerText, PlayerText_Posisiton, cv2.FONT_HERSHEY_TRIPLEX, 0.75, (255, 255, 255), 2)
                cv2.putText(frame, otherPlayer, otherPlayer_Text_Posisiton, cv2.FONT_HERSHEY_TRIPLEX, 0.75, (0,0,0), 2)
                cv2.imshow('21', frame)
                cv2.waitKey(5000) 

                gameSet = False
                player_turn = True
                cards_per_set +=1
                roundFin = False
                round2fin = True

            elif key == ord('y'):
                PlayerText = "Both Players : MORE CARDS PLEASE!"
                PlayerText_Posisiton = (75,350)
    
                text_size = cv2.getTextSize(PlayerText, cv2.FONT_HERSHEY_DUPLEX, 0.75, 2)[0]
                rect_position = (PlayerText_Posisiton[0], PlayerText_Posisiton[1] - text_size[1])
                rect_size = (text_size[0] + 30, text_size[1] + 10)
                cv2.rectangle(frame, rect_position, (rect_position[0] + rect_size[0], rect_position[1] + rect_size[1]), (255, 255, 255), -1)
    
                cv2.putText(frame, PlayerText, PlayerText_Posisiton, cv2.FONT_HERSHEY_TRIPLEX, 0.75, (200, 200, 200), 2)
                cv2.imshow('21', frame)
                cv2.waitKey(5000)
    
                gameSet = False
                player_turn = True
                cards_per_set += 1
                roundFin = False
                round2fin = True

        elif pC_score > 17 and pC_score <= 21 and p1_score <=19:
            key = cv2.waitKey(1)
            if key == ord('y'):
                PlayerText = "PLAYER 1 : MORE CARDS PLEASE!"
                otherPlayer = "COM : I'm gonna pass"
                PlayerText_Posisiton = (75,350)
                otherPlayer_Text_Posisiton = (75, 400)

                text_size = cv2.getTextSize(PlayerText, cv2.FONT_HERSHEY_DUPLEX, 0.75, 2)[0]
                rect_position = (PlayerText_Posisiton[0] - 5, PlayerText_Posisiton[1] - 5 - text_size[1])
                rect_size = (text_size[0] + 30, text_size[1] + 20)
                cv2.rectangle(frame, rect_position, (rect_position[0] + rect_size[0], rect_position[1] + rect_size[1]), (255,255,255), -1)

                text_size = cv2.getTextSize(otherPlayer, cv2.FONT_HERSHEY_DUPLEX, 0.75, 2)[0]
                rect_position = (otherPlayer_Text_Posisiton[0] - 5, otherPlayer_Text_Posisiton[1] - 5 - text_size[1])
                rect_size = (text_size[0] + 30, text_size[1] + 20)
                cv2.rectangle(frame, rect_position, (rect_position[0] + rect_size[0], rect_position[1] + rect_size[1]), (0, 0, 0), -1)

                cv2.putText(frame, PlayerText, PlayerText_Posisiton, cv2.FONT_HERSHEY_TRIPLEX, 0.75, (0,0,0), 2)
                cv2.putText(frame, otherPlayer, otherPlayer_Text_Posisiton, cv2.FONT_HERSHEY_TRIPLEX, 0.75, (255,255,255), 2)
                cv2.imshow('21', frame)
                cv2.waitKey(5000) 

                gameSet = False
                player_turn = True
                cards_per_set += 1
                roundFin = False
                round2fin = True
            elif key == ord('n'):
                gameSet = True
        else:
            gameSet = True

    if player_turn == False and gameSet == True and roundFin == True and round2fin == False:
        if p1_score <= 21 and (p1_score > pC_score or pC_score > 21):
            winner_text = "PLAYER 1 WIN! *(^v^)*" 

        elif pC_score <= 21 and (p1_score < pC_score or p1_score > 21):
            winner_text = "COM wins! *(>.<)*"
        
        elif pC_score > 21 and p1_score > 21:
            if pC_score < p1_score:
                winner_text = "COM wins! *(>.<)*"
            else : winner_text = "PLAYER 1 WIN! *(^v^)*"
        else :
            winner_text = "DRAW!! 0.0"
        
        text_size = cv2.getTextSize(winner_text, cv2.FONT_HERSHEY_DUPLEX, 0.75, 2)[0]
        winner_text_posisiton_x = int((frame.shape[1] - text_size[0]) / 2)
        winner_text_posisiton_y = int((frame.shape[0] + text_size[1]) / 2)

        scoreP1Text = f"P1 Score: {p1_score}"
        scorePCText = f"COM Score: {pC_score}"

        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), -1)
        cv2.putText(frame, winner_text, (winner_text_posisiton_x - 100, winner_text_posisiton_y), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 0, 0), 4)

        cv2.putText(frame, scoreP1Text, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)
        text_size = cv2.getTextSize(scoreP1Text, cv2.FONT_HERSHEY_DUPLEX, 0.75, 2)[0]
        rect_position = (8, 27 - text_size[1])
        rect_size = (text_size[0] + 30, text_size[1] + 10)
        cv2.rectangle(frame, rect_position, (rect_position[0] + rect_size[0], rect_position[1] + rect_size[1]), (200, 200, 200), 2)
        
    
        cv2.putText(frame, scorePCText, (400, 30), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 0), 2)
        text_size = cv2.getTextSize(scorePCText, cv2.FONT_HERSHEY_DUPLEX, 0.75, 2)[0]
        rect_position = (395, 27 - text_size[1])
        rect_size = (text_size[0] + 30, text_size[1] + 10)
        cv2.rectangle(frame, rect_position, (rect_position[0] + rect_size[0], rect_position[1] + rect_size[1]), (0, 0, 0), 2)

        for i in range(cards_per_set):
            if i < len(p1_cards):
                label = p1_cards[i]["label"]
                text = f"P1: {label}"
                cv2.putText(frame, text, (0, 100 + i * 30), cv2.FONT_HERSHEY_TRIPLEX, 0.75, (255, 255, 255), 2)

            if i < len(pC_cards):
                label = pC_cards[i]["label"]
                text = f"COM: {label}"
                cv2.putText(frame, text, (475, 100 + i * 30), cv2.FONT_HERSHEY_TRIPLEX, 0.75, (0, 0, 0), 2)

    somerules = "press 'r' for a rematch with me >:("
    cv2.putText(frame, somerules, (0, 475),cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,255), 1)
    cv2.imshow('21', frame)

    if cv2.waitKey(1) & 0xFF == ord('r'):
        cards_per_set = 2
        p1_cards = []
        pC_cards = []
        card_counter = 0
        player_turn = True  
        gameSet = False
        roundFin = False
        round2fin = False

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
