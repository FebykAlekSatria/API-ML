# DETECTION OOD
API Detection OOD berfungsi untuk melakukan klasifikasi kalimat (In-Domain) ID dan kalimat OOD(Out-Of-Domain). Deteksi ini berfungsi untuk mempermudah mengatasi permasalahan kalimat yang tidak bisa ditanggapi oleh chatbot.

| Fitur         	| Penggunaan     	| Parameter 	| Keterangan                                                                                                            	|
|---------------	|----------------	|-----------	|-----------------------------------------------------------------------------------------------------------------------	|
| Upload File   	| /upload        	| File      	| Mengambil dataset   dari inputan pengguna berupa data .csv                                                            	|
| preprocessing 	| /preprocessing 	| -         	| Melakukan processing   preprocessing yaitu case folding, clean punction, filtering, dan stemming                      	|
| tfidf         	| /tfidf         	| -         	| Melakukan pembobotan   kata                                                                                           	|
| best          	| /best          	| -         	| Melakukan traning   dengan alghoritma Support Vector Machine dan melakukan perbandingan kesetiap   kernel dan nilai C 	|
| train         	| /training      	| Kernel, C 	| Melakukan traning   dengan alghoritma Support Vector Machine                                                          	|
| Detection     	| /              	| Kalimat   	| Melakukan deteksi   kalimat masukan pengguna                                                                          	|

#Format Dataset
Dataset harus dengan format csv dengan kolom Kalimat dan kelas

| Kalimat                      	| Kelas 	|
|------------------------------	|-------	|
| temukan lokasi wisata pantai 	| OOD   	|
| Tampilkan KHS                	| ID    	|
