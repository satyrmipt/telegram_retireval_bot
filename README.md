# telegram_retireval_bot
Англоязычный (!) Телеграм-бор https://t.me/TBB_chat_bot использует retrieval-подход для ответа на вопросы в стиле Леонарда из "Теории большого взрыва".
Прежде чем отправлять сообщения боту, его нужно запустить. Бот игнорирует все сообщения, отправленные до его запуска. Для запуска перейдите в папку проекта и выполните команду:

  * python main.py

Будет запущен бот в конфигурации bi-encoder + cross-encoder

Чтобы убедиться, что бот работает, отправьте ему после запуска "hello" без кавычек: бот ответит на это сообщение не запуская механизма retrieval.

Для запуска необходимы пакеты, перечисленые в requirements.txt. Так же необходимо прописать путь к моим лисным модулям: my_tokenize_vectorize.py, my_bot.py или выложить их в папку, где их увидит python.

Некоторые из них я установил через conda forge.

Файлы проекта:

 * data_preparation.ipynb: сбор данных из интернета, изучение, обогащение
 * data_vectorization.ipynb: обогащение данных векторизованными представлениями вопросов и ответов, использованы два подхода: glove-вкторизация (без дообучения посредством mittens) и Sentence BERT векоризация (взят предобученный энкодер и дообучен на triple loss)
 * bi_enc_cross_enc_fine_tuning.ipynb: ноутбку с дообучением модели BI-энкодера и Cross-энкодера.
 * my_tokenize_vectorize.py: мой модуль, содержит класс токенайзера и два класса векторизации (glove и SBERT)
 * my_bot.py: мой модуль, содержит инстурменты поиска ответа на вопрос
 * main.py: исполняемый файл проекта, описывает устройство телеграм-бота.


