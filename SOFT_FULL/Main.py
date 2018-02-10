import knn_procedure
import video_procedure as vproc
import frame_processing as fproc

digits_file = "digits.png"
video_folder = "videos/"
video_files = [
    video_folder + "video-0.avi",
    video_folder + "video-1.avi",
    video_folder + "video-2.avi",
    video_folder + "video-3.avi",
    video_folder + "video-4.avi",
    video_folder + "video-5.avi",
    video_folder + "video-6.avi",
    video_folder + "video-7.avi",
    video_folder + "video-8.avi",
    video_folder + "video-9.avi"
]

knn_procedure.init_knn(digits_file)

outfile = open('out.txt', 'w')
outfile.write('RA 244/2015 Nena Vidovic\n')
outfile.write('file\tsum\n')

for video_file in video_files:
    vproc.video_processing(video_file)
    vf = video_file.replace("videos/","")
    outfile.write(vf + "\t" + str(fproc.sum_of_numbers) + "\n")
    fproc.reset_information()

outfile.close()

import test
