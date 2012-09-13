#ifndef SIMPLE_PROGRESSBARS_FOR_VIEWING_STATUS_OF_COMPUTATIONS
#define SIMPLE_PROGRESSBARS_FOR_VIEWING_STATUS_OF_COMPUTATIONS

#define PROGRESSBAR_MANAGEMENT           \
unsigned progressbar_current_length      \
       , progressbar_total_length        \
       , progressbar_event_status

#define PROGRESSBAR_INITIALIZE(title)                          \
do {                                                           \
  std::string progressbar_title(title);                        \
  progressbar_total_length = 94 - progressbar_title.length();  \
  progressbar_current_length = progressbar_event_status = 0;   \
  (std::cout << progressbar_title << "..").flush();            \
}while(0)

#define PROGRESSBAR_STEPDISPLAY(statuslength)                           \
  while(float(progressbar_event_status)/(statuslength)                  \
       > float(progressbar_current_length)/progressbar_total_length) {  \
    ++progressbar_current_length;                                       \
    (std::cout << ".").flush();                                         \
  }++progressbar_event_status


#endif