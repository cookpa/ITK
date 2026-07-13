/*=========================================================================
 *
 *  Copyright NumFOCUS
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         https://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#include "gtest/gtest.h"
#include "itkGiplImageIO.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkVector.h"

#include <filesystem>
#include <fstream>
#include <sstream>

#define _STRING(s) #s
#define TOSTRING(s) std::string(_STRING(s))

namespace
{
using ScalarImageType = itk::Image<unsigned char, 2>;

ScalarImageType::Pointer
MakeScalarImage(unsigned int side)
{
  auto                            image = ScalarImageType::New();
  const ScalarImageType::SizeType size{ { side, side } };
  image->SetRegions(ScalarImageType::RegionType(size));
  image->Allocate();
  image->FillBuffer(42);
  return image;
}

std::string
OutputPath(const std::string & name)
{
  return TOSTRING(ITK_TEST_OUTPUT_DIR) + "/" + name;
}
} // namespace

TEST(GiplImageIO, WriteOfUnsupportedComponentTypePreservesExistingFile)
{
  const std::string path = OutputPath("gipl_b53_existing.gipl");
  const std::string originalContents = "pre-existing data that must survive a failed Gipl write";
  {
    std::ofstream out(path, std::ios::binary);
    out << originalContents;
  }

  using UnsupportedImageType = itk::Image<unsigned long, 2>;
  auto                                 image = UnsupportedImageType::New();
  const UnsupportedImageType::SizeType size{ { 2, 2 } };
  image->SetRegions(UnsupportedImageType::RegionType(size));
  image->Allocate();
  image->FillBuffer(7);

  auto giplIO = itk::GiplImageIO::New();
  using WriterType = itk::ImageFileWriter<UnsupportedImageType>;
  auto writer = WriterType::New();
  writer->SetImageIO(giplIO);
  writer->SetInput(image);
  writer->SetFileName(path);

  EXPECT_THROW(writer->Update(), itk::ExceptionObject);

  std::ifstream      in(path, std::ios::binary);
  std::ostringstream contents;
  contents << in.rdbuf();
  EXPECT_EQ(contents.str(), originalContents);
}

TEST(GiplImageIO, ReadOfTruncatedUncompressedFileThrows)
{
  const std::string path = OutputPath("gipl_b54_truncated.gipl");
  auto              image = MakeScalarImage(8);

  auto writerIO = itk::GiplImageIO::New();
  using WriterType = itk::ImageFileWriter<ScalarImageType>;
  auto writer = WriterType::New();
  writer->SetImageIO(writerIO);
  writer->SetInput(image);
  writer->SetFileName(path);
  ASSERT_NO_THROW(writer->Update());

  const auto fullSize = std::filesystem::file_size(path);
  ASSERT_GT(fullSize, 20u);
  std::filesystem::resize_file(path, fullSize - 10);

  auto readerIO = itk::GiplImageIO::New();
  using ReaderType = itk::ImageFileReader<ScalarImageType>;
  auto reader = ReaderType::New();
  reader->SetImageIO(readerIO);
  reader->SetFileName(path);

  EXPECT_THROW(reader->Update(), itk::ExceptionObject);
}

TEST(GiplImageIO, ReadOfTruncatedCompressedFileThrows)
{
  const std::string path = OutputPath("gipl_b55_truncated.gipl.gz");
  auto              image = MakeScalarImage(8);

  auto writerIO = itk::GiplImageIO::New();
  using WriterType = itk::ImageFileWriter<ScalarImageType>;
  auto writer = WriterType::New();
  writer->SetImageIO(writerIO);
  writer->SetInput(image);
  writer->SetFileName(path);
  ASSERT_NO_THROW(writer->Update());

  const auto fullSize = std::filesystem::file_size(path);
  ASSERT_GT(fullSize, 20u);
  std::filesystem::resize_file(path, fullSize / 2);

  auto readerIO = itk::GiplImageIO::New();
  using ReaderType = itk::ImageFileReader<ScalarImageType>;
  auto reader = ReaderType::New();
  reader->SetImageIO(readerIO);
  reader->SetFileName(path);

  EXPECT_THROW(reader->Update(), itk::ExceptionObject);
}

TEST(GiplImageIO, WriteOfMultiComponentImageThrows)
{
  using VectorPixelType = itk::Vector<float, 3>;
  using VectorImageType = itk::Image<VectorPixelType, 2>;

  auto                            image = VectorImageType::New();
  const VectorImageType::SizeType size{ { 4, 4 } };
  image->SetRegions(VectorImageType::RegionType(size));
  image->Allocate();
  image->FillBuffer(VectorPixelType({ 1.0f, 2.0f, 3.0f }));

  const std::string path = OutputPath("gipl_b56_vector.gipl");

  auto giplIO = itk::GiplImageIO::New();
  using WriterType = itk::ImageFileWriter<VectorImageType>;
  auto writer = WriterType::New();
  writer->SetImageIO(giplIO);
  writer->SetInput(image);
  writer->SetFileName(path);

  EXPECT_THROW(writer->Update(), itk::ExceptionObject);
}

TEST(GiplImageIO, RoundTripsIntAndUnsignedIntComponentTypes)
{
  using IntImageType = itk::Image<int, 2>;
  using UIntImageType = itk::Image<unsigned int, 2>;
  const IntImageType::SizeType size{ { 3, 3 } };

  auto intImage = IntImageType::New();
  intImage->SetRegions(IntImageType::RegionType(size));
  intImage->Allocate();
  intImage->FillBuffer(-70000);
  intImage->GetBufferPointer()[1] = 123456;

  auto uintImage = UIntImageType::New();
  uintImage->SetRegions(UIntImageType::RegionType(size));
  uintImage->Allocate();
  uintImage->FillBuffer(3000000000u);
  uintImage->GetBufferPointer()[1] = 654321u;

  const std::string intPath = OutputPath("gipl_enh_int.gipl");
  const std::string uintPath = OutputPath("gipl_enh_uint.gipl");

  auto intWriter = itk::ImageFileWriter<IntImageType>::New();
  intWriter->SetImageIO(itk::GiplImageIO::New());
  intWriter->SetInput(intImage);
  intWriter->SetFileName(intPath);
  ASSERT_NO_THROW(intWriter->Update());

  auto uintWriter = itk::ImageFileWriter<UIntImageType>::New();
  uintWriter->SetImageIO(itk::GiplImageIO::New());
  uintWriter->SetInput(uintImage);
  uintWriter->SetFileName(uintPath);
  ASSERT_NO_THROW(uintWriter->Update());

  auto intReader = itk::ImageFileReader<IntImageType>::New();
  intReader->SetImageIO(itk::GiplImageIO::New());
  intReader->SetFileName(intPath);
  ASSERT_NO_THROW(intReader->Update());

  auto uintReader = itk::ImageFileReader<UIntImageType>::New();
  uintReader->SetImageIO(itk::GiplImageIO::New());
  uintReader->SetFileName(uintPath);
  ASSERT_NO_THROW(uintReader->Update());

  const auto numberOfPixels = intImage->GetPixelContainer()->Size();
  for (unsigned int i = 0; i < numberOfPixels; ++i)
  {
    EXPECT_EQ(intImage->GetBufferPointer()[i], intReader->GetOutput()->GetBufferPointer()[i]);
    EXPECT_EQ(uintImage->GetBufferPointer()[i], uintReader->GetOutput()->GetBufferPointer()[i]);
  }
}
